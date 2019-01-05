from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import logging.config
import sys
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import numpy as np
import reader
import utils

CUDA_VISIBLE_DEVICES = 1

info = ""
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_fp = '{0}.log'.format(f'{info}model')
file_handler = logging.FileHandler(log_fp)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

root_path = 'data/tweet/file'
EMBEDDING_DIM = 50
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_CLASS = 2
PREDICT_STEPS = 5
MAX_GRAD_NORM = 15
NUM_EPOCH = 10
LINEAR_DIM = 64
DECAY_STEP = 10
DECAY_RATE = 0.98
STOCK_SIZE = 87
LAMBDA = 0.01
FEATURE_NUM = 3


class StockMovementPrediction(object):
    def __init__(self, is_training, batch_size, num_steps, linear_dim, num_head, drop_out,
                 max_num_news, max_num_words, lr, vocab_size):
        self.linear_dim = linear_dim
        self.num_head = num_head
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.is_training = is_training
        self.drop_out = drop_out
        self.max_num_news = max_num_news
        self.max_num_words = max_num_words
        self.lr = lr
        self.hidden_size = HIDDEN_SIZE
        self.attention_reg = 0
        self.final_state = None
        self.initial_state = None

        assert self.linear_dim % self.num_head == 0

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(tf.float32, [None, self.num_steps, FEATURE_NUM], name="price")
            self.targets = tf.placeholder(tf.int64, [None, PREDICT_STEPS], name='label')
            # self.step_targets = tf.placeholder(tf.int64, [None, self.num_steps], name='step_label')
            self.news_ph = tf.placeholder(tf.int64, [None, self.num_steps, self.max_num_news, self.max_num_words], name='news')
            self.word_table_init = tf.placeholder(tf.float32, [vocab_size, EMBEDDING_DIM], name='word_embedding')

        with tf.name_scope('word_embeddings'):
            with tf.variable_scope('embeds'):
                word_table = tf.get_variable('word_table', initializer=self.word_table_init, trainable=False)
                self.news = tf.nn.embedding_lookup(word_table, self.news_ph, name='news_word_embeds')

        logger.info(
            f"embedding_size:{EMBEDDING_DIM}, max_num_news:{self.max_num_news}, max_num_words:{self.max_num_words},"
            f" lr:{self.lr}, batch_size:{self.batch_size}, num_head:{self.num_head}, drop_out:{self.drop_out},"
            f" num_step:{self.num_steps}")

        outputs = self.encode()
        logits = self.decode(outputs, self.final_state)
        with tf.name_scope("loss_function"):
            cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.targets, logits=logits)
            parameter_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 0.001
            self.loss = (1-LAMBDA) * cross_entropy_loss + LAMBDA * (self.attention_reg + parameter_loss)
            self.prediction = tf.argmax(logits, -1)

        if not self.is_training:
            return

        # Optimizer #
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(self.lr, global_step, DECAY_STEP, DECAY_RATE, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

    def encode(self):
        """
        :return:
        """
        encoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size)
        if self.is_training:
            encoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size, dropout_keep_prob=1 - self.drop_out)
        cell = tf.nn.rnn_cell.MultiRNNCell([encoder_cell] * NUM_LAYERS)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        outputs = []
        state = self.initial_state

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            for time_step in range(self.num_steps):
                with tf.variable_scope("word_level_self_attention", reuse=tf.AUTO_REUSE):
                    daily_news = []
                    for news_num in range(self.max_num_news):
                        # [batch_size, max_sequence, num_head, embedding]
                        one_news = self._multi_head_self_attention(self.news[:, time_step, news_num, :, :],
                                                                   self.num_head, self.linear_dim, self.drop_out)
                        att_one_news = tf.reshape(one_news,
                                                 [self.batch_size, self.linear_dim])  # [batch_size, linear_dim]
                        daily_news.append(att_one_news)
                    daily_news = tf.transpose(daily_news, [1, 0, 2])    # [batch_size, max_num_news, linear_dim]
                with tf.variable_scope('news_level_self_attention', reuse=tf.AUTO_REUSE):
                    daily_news = self._multi_head_self_attention(daily_news, self.num_head, self.linear_dim, self.drop_out)
                    att_daily_news = self._single_attention(daily_news, state[0].h)
                price = tf.layers.dense(self.input_data[:, time_step, :], self.linear_dim//self.num_head)
                cell_input = tf.concat([price, att_daily_news], -1)
                cell_output, state = cell(cell_input, state)
                outputs.append(cell_output)
        self.final_state = state
        return outputs

    def decode(self, encode, state):
        """
        multi step stock movement prediction
        encode： encode hidden states
        state: final hidden states of encoder
        return the predicted logits
        """
        with tf.variable_scope("decoder"):
            # [batch_size, max_time, num_units]
            attention_states = tf.transpose(encode, [1, 0, 2])
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.hidden_size, attention_states)

            decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size)
            if self.is_training:
                decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size,
                                                                     dropout_keep_prob=1 - self.drop_out)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=self.hidden_size)

            helper = tf.contrib.seq2seq.TrainingHelper(encode, [PREDICT_STEPS for _ in range(self.batch_size)],
                                                       time_major=True)
            decoder_initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=state[0])
            projection_layer = layers_core.Dense(units=NUM_CLASS, use_bias=False, activation=tf.nn.sigmoid)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                                      output_layer=projection_layer)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = outputs.rnn_output
        return logits

    def _multi_head_self_attention(self, v, num_head, dim, drop_out):
        with tf.name_scope('multi_head_self_attention'):
            # linear projection
            with tf.variable_scope('linear_projection'):
                vp_q = tf.layers.dense(v, dim, use_bias=False)
                vp_k = tf.layers.dense(v, dim, use_bias=False)
                vp_v = tf.layers.dense(v, dim, use_bias=False)
            # split_heads
            with tf.variable_scope('split_head'):
                def split_last_dimension_then_transpose(tensor, num_head, dim):
                    t_shape = tensor.get_shape().as_list()
                    tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_head, dim])
                    return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, dim]

                vs_q = split_last_dimension_then_transpose(vp_q, num_head, dim // num_head)
                vs_k = split_last_dimension_then_transpose(vp_k, num_head, dim // num_head)
                vs_v = split_last_dimension_then_transpose(vp_v, num_head, dim // num_head)
            depth = dim // num_head
            vs_q *= depth ** -0.5

            # mask the padding vec
            # masked = [[0 if vec_sum == 0 else 1 for vec_sum in batch] for batch in tf.reduce_sum(v, axis=-1)]
            masked = tf.map_fn(lambda batch: tf.map_fn(lambda x: 0.0 if x == 0.0 else 1.0, batch), tf.reduce_sum(v, axis=-1))
            bias = utils.get_padding_bias(masked)
            # scaled_dot_product
            with tf.variable_scope('scaled_dot_product'):
                logits = tf.matmul(vs_q, vs_k, transpose_b=True)
                logits += bias
                weights = tf.nn.softmax(logits, name="attention_softmax")
                reg = tf.matmul(weights, weights, transpose_b=True) - tf.eye(num_head, batch_shape=[self.batch_size])
                self.attention_reg += tf.reduce_sum(tf.norm(reg, axis=[1, 2]))
                if self.is_training:
                    weights = tf.nn.dropout(weights, 1.0 - drop_out)
                attention_output = tf.matmul(weights, vs_v)  # [batch_size, num_head, max_sequence_len, dim]
                # [batch_size, max_sequence_len, num_head, dim]
                attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        pooled = self._pooling(attention_output)  # [batch_size, num_head, linear_dim//num_head]

        return pooled

    def _single_attention(self, v, k):
        v_pro = tf.layers.dense(v, self.hidden_size, name='v', activation=tf.nn.tanh)
        k_ext = tf.tile(k, [1, self.num_head])  # [batch_size, num_head * hidden_size]
        k_ext = tf.reshape(k_ext, [self.batch_size, self.num_head, self.hidden_size])
        att = tf.reduce_sum(tf.multiply(v_pro, k_ext), axis=-1)
        att = tf.nn.softmax(att)    # [batch_size, num_head]
        tf.summary.histogram('aspect_attention', att)
        attention_output = tf.reduce_sum(v * tf.expand_dims(att, -1), 1)  # [batch_size, dim]

        if self.is_training:
            attention_output = tf.nn.dropout(attention_output, 1-self.drop_out)
        return attention_output

    @staticmethod
    def _pooling(v):
        return tf.reduce_max(v, axis=-3)


def run_epoch(session, merged, model, data, train_op, flag, output_log):
    total_costs = []
    iters = 0
    total_auc = []
    all_acc = [[] for _ in range(PREDICT_STEPS+1)]
    mcc = []
    state = session.run(model.initial_state)
    for x, y, _, news, _ in reader.news_iterator(data, model.batch_size, model.num_steps,
                                                 model.max_num_news, model.max_num_words, flag):
        cost, summary, state, _, prediction = session.run(
            [model.loss, merged, model.final_state, train_op, model.prediction],
            {model.input_data: x, model.targets: y, model.news_ph: news, model.initial_state: state})
        total_costs.append(cost)
        try:
            total_auc.append(roc_auc_score(y_true=y.reshape(-1), y_score=prediction.reshape(-1)))
        except ValueError:
            print("auc error")
        iters += model.num_steps
        for i in range(PREDICT_STEPS):
            all_acc[i].append(accuracy_score(prediction[:, i], y[:, i]))
        all_acc[-1].append(accuracy_score(prediction.reshape(-1), y.reshape(-1)))
        mcc.append(matthews_corrcoef(y_true=y.reshape(-1), y_pred=prediction.reshape(-1)))
        if output_log and iters % 500 == 0:
            logger.info("After %d steps, cost is %.5f acc %.5f auc %.5f mcc %.5f" % (
                iters, np.mean(total_costs), np.mean(all_acc[0]), np.mean(total_auc), np.mean(mcc)))
            if iters % 2000 == 0:
                print(f"prediction {prediction}, label {y}")

    return np.mean(total_costs), np.mean(all_acc, axis=1), summary, np.mean(total_auc), np.mean(mcc)


def tuning_parameter():
    for batch_size in [4, 8, 16, 32]:
        for num_head in [4, 8, 2]:
            for max_num_news in [30, 20, 10]:
                for max_num_words in [40, 20, 20, 10]:
                    for lr in [0.1, 0.01, 0.001]:
                        for drop_out in [0.3, 0.5, 0.1]:
                            for num_step in [10, 15, 20, 5]:
                                yield max_num_news, max_num_words, lr, batch_size, num_head, drop_out, num_step


def main(_):
    train_data, valid_data, test_data = reader.news_raw_data()
    word_table_init, vocab_size = reader.init_word_table()
    parameter_gen = tuning_parameter()
    while True:
        try:
            max_num_news, max_num_words, lr, batch_size, num_head, drop_out, num_steps = next(parameter_gen)
        except StopIteration:
            break
        initializer = tf.random_uniform_initializer(-0.01, 0.01)
        tf.reset_default_graph()
        if os.path.exists(os.path.join(root_path, f"{info}train_preprocess.pkl")):
            os.remove(os.path.join(root_path, f'{info}train_preprocess.pkl'))
        if os.path.exists(os.path.join(root_path, f"{info}valid_preprocess.pkl")):
            os.remove(os.path.join(root_path, f'{info}valid_preprocess.pkl'))
        if os.path.exists(os.path.join(root_path, f"{info}test_preprocess.pkl")):
            os.remove(os.path.join(root_path, f'{info}test_preprocess.pkl'))
        with tf.name_scope("Train"):
            with tf.variable_scope("StockMovementPrediction", reuse=None, initializer=initializer):
                train_model = StockMovementPrediction(True, batch_size, num_steps, LINEAR_DIM, num_head, drop_out,
                                                      max_num_news, max_num_words, lr, vocab_size)
        saver = tf.train.Saver()
        sum_path = 'tensorboard/%smodel_batch%d_h%d_d%.2f_step%d_news%d_words%d_lr%.5f' % \
                    (info, batch_size, num_head, drop_out, num_steps, max_num_news, max_num_words, lr)

        writer = tf.summary.FileWriter(sum_path, tf.Session().graph)
        merged = tf.summary.merge_all()
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with tf.Session(config=config) as session:
            session.run(tf.initializers.global_variables(), feed_dict={train_model.word_table_init: word_table_init})
            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters += dim.value
                total_parameters += variable_parameters
            logger.info("total parameters: %d", total_parameters)
            for i in range(NUM_EPOCH):
                logger.info("In iteration: %d" % (i + 1))
                train_cost, acc, summary, auc, mcc = run_epoch(session, merged, train_model, train_data,
                                                               train_model.train_op, 'train', True)
                logger.info("Epoch: %d Training average Cost: %.5f auc is %.5f, mcc is %.5f" %
                            (i + 1, train_cost, auc, mcc))
                writer.add_summary(summary, i)
                valid_cost, acc, _, auc, mcc = run_epoch(session, merged, train_model, valid_data,
                                                         tf.no_op(), 'valid', False)
                logger.info("Epoch: %d Validation Cost: %.5f, auc is %.5f mcc is %.5f" %
                            (i + 1, valid_cost, auc, mcc))
                for j in range(PREDICT_STEPS + 1):
                    logger.info("predict step %d acc: %.5f", j, acc[j])
            test_cost, acc, _, auc, mcc = run_epoch(session, merged, train_model, test_data,
                                                    tf.no_op(), 'test', False)
            logger.info("Test Cost: %.3f, auc is %.5f, mcc is %.5f" % (test_cost, auc, mcc))
            for j in range(PREDICT_STEPS + 1):
                logger.info("predict step %d acc: %.5f", j, acc[j])
            saver.save(session, 'model_saver/%smodel_batch%d_h%d_d%.2f_step%d_news%d_words%d_lr%.5f.ckpt' %
                       (info, batch_size, num_head, drop_out, num_steps, max_num_news, max_num_words, lr))


if __name__ == "__main__":
    tf.app.run()
