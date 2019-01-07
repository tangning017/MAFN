from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import logging.config
import sys
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
PREDICT_STEPS = 1
NUM_EPOCH = 20
LINEAR_DIM = 64
DECAY_STEP = 10
DECAY_RATE = 0.98
STOCK_SIZE = 87
FEATURE_NUM = 3


class StockMovementPrediction(object):
    def __init__(self, is_training, batch_size, num_steps, linear_dim, num_head, drop_out,
                 max_num_news, max_num_words, lr, vocab_size, att_lambda, param_lambda):
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
        self.att_lambda = att_lambda
        self.param_lambda = param_lambda
        self.attention_reg = []
        self.final_state = None
        self.initial_state = None

        assert self.linear_dim % self.num_head == 0

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(tf.float32, [None, self.num_steps, FEATURE_NUM], name="price")
            self.targets = tf.placeholder(tf.float32, [None, PREDICT_STEPS], name='rate')
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
            f" num_step:{self.num_steps}, att_lambda: {self.att_lambda}, param_lambda: {self.param_lambda}")

        outputs = self.encode()
        logits = self.decode(outputs, self.final_state)
        with tf.name_scope("loss_function"):
            # self.cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.targets, logits=logits)
            self.mse_loss = tf.losses.mean_squared_error(labels=self.targets, predictions=tf.squeeze(logits))
            self.rmse_loss = tf.sqrt(self.mse_loss)
            trainable_vars = tf.trainable_variables()
            self.param_loss = self.param_lambda * tf.reduce_mean([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name])
            self.att_loss = self.att_lambda * tf.reduce_mean(self.attention_reg)
            self.loss = self.rmse_loss + self.att_loss + self.param_loss
            self.prediction = logits

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
        # encoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size)
        encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        # if self.is_training:
        #     encoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size, dropout_keep_prob=1 - self.drop_out)
        cell = tf.nn.rnn_cell.MultiRNNCell([encoder_cell] * NUM_LAYERS)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        outputs = []
        state = self.initial_state

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            for time_step in range(self.num_steps):
                daily_news = []
                with tf.variable_scope("word_level_bidirectional_lstm"):
                    for num_news in range(self.max_num_news):
                        cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                        cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                        bioutput, _ = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw, cell_bw, self.news[:, time_step, num_news, :, :],
                            dtype=tf.float32, time_major=False)
                        bilstm_output = (bioutput[0] + bioutput[1]) / 2
                        bilstm_output = tf.reduce_mean(bilstm_output, axis=-2)
                        daily_news.append(bilstm_output)
                daily_news = tf.transpose(daily_news, [1, 0, 2])
                with tf.variable_scope('news_level_self_attention', reuse=tf.AUTO_REUSE):
                    att_daily_news = self._multi_head_self_attention(daily_news, self.num_head, self.linear_dim, self.drop_out)
                    # att_daily_news = tf.concat([daily_news, att_daily_news], -1)
                    att_daily_news = self._single_attention(att_daily_news, state[0].h)
                price = tf.layers.dense(self.input_data[:, time_step, :],
                                        self.linear_dim//self.num_head, activation=tf.nn.selu)
                cell_input = tf.concat([price, att_daily_news], -1)
                # cell_input = price
                # cell_input = att_daily_news
                # cell_input = tf.layers.dense(cell_input, self.hidden_size)
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

            # decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size)
            decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            # if self.is_training:
            #     decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size,
            #                                                          dropout_keep_prob=1 - self.drop_out)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=self.hidden_size)

            helper = tf.contrib.seq2seq.TrainingHelper(encode, [PREDICT_STEPS for _ in range(self.batch_size)],
                                                       time_major=True)
            decoder_initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=state[0])
            projection_layer = layers_core.Dense(units=1, activation=tf.nn.selu)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                                      output_layer=projection_layer)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = outputs.rnn_output
        return logits

    def _multi_head_self_attention(self, v, num_head, dim, drop_out):
        with tf.name_scope('multi_head_self_attention'):
            # linear projection
            with tf.variable_scope('linear_projection'):
                vp_q = tf.layers.dense(v, dim, use_bias=False, activation=tf.nn.selu)
                vp_k = tf.layers.dense(v, dim, use_bias=False, activation=tf.nn.selu)
                vp_v = tf.layers.dense(v, dim, use_bias=False, activation=tf.nn.selu)
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
                if self.is_training:
                    weights = tf.nn.dropout(weights, 1.0 - drop_out)
                attention_output = tf.matmul(weights, vs_v)  # [batch_size, num_head, max_sequence_len, dim]
                # [batch_size, max_sequence_len, num_head, dim]
                attention_output = tf.transpose(attention_output, [0, 2, 1, 3])

                def add_multi_head_self_attention_reg(weights, batch_size, num_head):
                    t_shape = weights.get_shape().as_list()
                    # [batch_size, max_sequence_len, num_head, max_sequence_len]
                    tensor = tf.transpose(weights, [0, 2, 1, 3])
                    reg = tf.matmul(tensor, tensor, transpose_b=True) - \
                        tf.eye(num_head, batch_shape=[batch_size, t_shape[-1]])
                    return tf.reduce_sum(tf.norm(reg, axis=[-2, -1]))

                self.attention_reg += [add_multi_head_self_attention_reg(weights, self.batch_size, num_head)]

        pooled = self._pooling(attention_output)  # [batch_size, max_sequence, linear_dim//num_head]

        return pooled

    def _single_attention(self, v, k):
        v_pro = tf.layers.dense(v, self.hidden_size, name='v', activation=tf.nn.tanh)
        k_ext = tf.tile(k, [1, self.num_head])  # [batch_size, num_head * hidden_size]
        k_ext = tf.reshape(k_ext, [self.batch_size, self.num_head, self.hidden_size])
        att = tf.reduce_sum(tf.multiply(v_pro, k_ext), axis=-1)
        att = tf.nn.softmax(att)    # [batch_size, num_news]
        tf.summary.histogram('aspect_attention', att)
        attention_output = tf.reduce_sum(v * tf.expand_dims(att, -1), 1)  # [batch_size, dim]
        return attention_output

    @staticmethod
    def _pooling(v):
        return tf.reduce_max(v, axis=-3)


def run_epoch(session, merged, model, data, flag, output_log):
    total_costs = []
    att_costs = []
    param_costs = []
    iters = 0
    state = session.run(model.initial_state)
    for x, y, _, news, _ in reader.news_iterator(data, model.batch_size, model.num_steps,
                                                 model.max_num_news, model.max_num_words, flag):
        cost, att_loss, param_loss, summary, state, prediction = session.run(
            [model.loss, model.att_loss, model.param_loss, merged, model.final_state, model.prediction],
            {model.input_data: x, model.targets: y, model.news_ph: news, model.initial_state: state})
        total_costs.append(cost)
        att_costs.append(att_loss)
        param_costs.append(param_loss)
        iters += model.num_steps
        if output_log and iters % 100 == 0:
            logger.info("After %d steps, cost is %.5f att cost is %.5f param cost is %.5f"
                        % (iters, cost, att_loss, param_loss))

    print(f"prediction {prediction}, label {y}")

    return np.mean(total_costs), np.mean(att_costs), np.mean(param_costs), summary


def tuning_parameter():
    # for max_num_news in [10]:
    #     for max_num_words in [20]:
    #         for lr in [0.01]:
    #             for drop_out in [0.3]:
    # for batch_size in [64]:
    #     for num_head in [4]:
    for num_step in [5]:
        for att_lambda in [0.0000]:
            for param_lambda in [0.00]:
                yield 10, 20, 0.01, 64, 4, 0.0, num_step, att_lambda, param_lambda


def main(_):
    train_data, valid_data, test_data = reader.news_raw_data()
    word_table_init, vocab_size = reader.init_word_table()
    parameter_gen = tuning_parameter()
    while True:
        try:
            max_num_news, max_num_words, lr, batch_size, num_head, drop_out, num_steps,\
                att_lambda, param_lambda = next(parameter_gen)
        except StopIteration:
            break
        # initializer = tf.random_uniform_initializer(-0.05, 0.05)
        initializer = tf.variance_scaling_initializer(scale=1.0, mode='fan_in')
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
                                                      max_num_news, max_num_words, lr, vocab_size,
                                                      att_lambda, param_lambda)
        saver = tf.train.Saver()
        sum_path = 'tensorboard/%smodel_batch%d_h%d_d%.2f_step%d_news%d_words%d_lr%.5f'\
                   % (info, batch_size, num_head, drop_out, num_steps, max_num_news, max_num_words, lr)

        writer = tf.summary.FileWriter(sum_path, tf.Session().graph)
        merged = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.3
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
                train_cost, att_cost, param_cost, summary = \
                    run_epoch(session, merged, train_model, train_data, 'train', True)
                logger.info("Epoch: %d Training average Cost: %.5f %.5f %.5f" %
                            (i + 1, train_cost, att_cost, param_cost))
                writer.add_summary(summary, i)
                valid_cost, att_cost, param_cost, _ = \
                    run_epoch(session, merged, train_model, valid_data, 'valid', False)
                logger.info("Epoch: %d Validation Cost: %.5f %.5f %.5f" %
                            (i + 1, valid_cost, att_cost, param_cost))
            test_cost, att_cost, param_cost, _ = run_epoch(session, merged, train_model, test_data, 'test', False)
            logger.info("Test Cost: %.3f %.3f %.3f" % (test_cost, att_cost, param_cost))
            saver.save(session, 'model_saver/%smodel_batch%d_h%d_d%.2f_step%d_news%d_words%d_lr%.5f.ckpt' %
                       (info, batch_size, num_head, drop_out, num_steps, max_num_news, max_num_words, lr))


if __name__ == "__main__":
    tf.app.run()
