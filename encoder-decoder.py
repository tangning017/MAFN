from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging.config
import sys
import numpy as np
import reader
from utils import eval_res

dataset_flag = 'news'
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_fp = '{0}.log'.format(f'{dataset_flag}_model')
file_handler = logging.FileHandler(log_fp)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class StockMovementPrediction(object):
    def __init__(self, config):
        self.linear_dim = config['linear_dim']
        self.num_head = config['num_head']
        self.batch_size = config['batch_size']
        self.num_steps = config['num_steps']
        self.drop_out = config['drop_out']
        self.max_num_news = config['max_num_news']
        self.max_num_words = config['max_num_words']
        self.feature_num = config['feature_num']
        self.predict_steps = config['predict_steps']
        self.embed_size = config['embed_size']
        self.lr = config['lr']
        self.hidden_size = config['hidden_size']
        self.att_lambda = config['att_lambda']
        self.param_lambda = config['param_lambda']
        self.vocab_size = config['vocab_size']
        self.classifregress = config['classifregress']
        self.attention_reg = []
        self.final_state = None
        self.initial_state = None

        assert self.linear_dim % self.num_head == 0
        if self.classifregress == 'regress':
            self.num_class = 1
        else:
            self.num_class = 2

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(tf.float32, [self.batch_size, self.num_steps, self.feature_num], name="price")
            self.targets = tf.placeholder(tf.float32, [self.batch_size, self.predict_steps], name='rate')
            self.news_ph = tf.placeholder(tf.int64, [self.batch_size, self.num_steps, self.max_num_news,
                                                     self.max_num_words], name='news')
            self.word_table_init = tf.placeholder(tf.float32, [self.vocab_size, self.embed_size], name='word_embedding')
            self.is_training = tf.placeholder(tf.bool, shape=(), name="train")
        with tf.name_scope('word_embeddings'):
            with tf.variable_scope('embeds'):
                word_table = tf.get_variable('word_table', initializer=self.word_table_init, trainable=False)
                self.news = tf.nn.embedding_lookup(word_table, self.news_ph, name='news_word_embeds')
                if self.embed_size > 50:
                    self.news = tf.layers.dense(self.news, 50, use_bias=False) # reduce the dimension of words

        logger.info(
            f"embedding_size:{self.embed_size}, max_num_news:{self.max_num_news}, max_num_words:{self.max_num_words},"
            f" lr:{self.lr}, batch_size:{self.batch_size}, num_head:{self.num_head}, drop_out:{self.drop_out},"
            f" num_step:{self.num_steps}, att_lambda: {self.att_lambda}, param_lambda: {self.param_lambda}")

        outputs = self.encode()
        pred = self.decode(outputs)
        if self.classifregress == 'regress':
            self.loss = self.mse_loss(pred, self.targets)
            self.logit = pred
        else:
            self.loss = self.cross_entropy(pred, self.targets)
            self.logit = tf.argmax(pred, axis=-1)

        trainable_vars = tf.trainable_variables()
        self.loss += self.param_lambda * tf.reduce_mean([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name])

        if self.is_training is None:
            return
        # Optimizer #
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr, global_step, 10, 0.96, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

    def encode(self):
        """
        :return:
        """
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        outputs = []
        state = self.initial_state

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            for time_step in range(self.num_steps):
                with tf.variable_scope("word_level_bidirectional_lstm"):
                    one_day_news = self.news[:, time_step, :, :, :]
                    one_day_news = tf.reshape(one_day_news,
                                               [self.batch_size*self.max_num_news, self.max_num_words, -1])
                    cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                    cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                    bioutput, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, one_day_news, dtype=tf.float32)
                    bilstm_output = tf.concat(bioutput, 2)
                    bilstm_output = tf.reduce_mean(bilstm_output, axis=-2)
                    daily_news = tf.reshape(bilstm_output, [self.batch_size, self.max_num_news, -1])
                with tf.variable_scope('news_level_attention'):
                    att_daily_news = self._multi_head_transform(daily_news, self.num_head, self.linear_dim)
                with tf.variable_scope("aspect_level_attention"):
                    att_daily_aspect, alpha = self._single_attention(att_daily_news, state.h)
                cell_input = att_daily_aspect
                cell_output, state = cell(cell_input, state)
                outputs.append(cell_output)
        self.final_state = state
        return outputs

    def decode(self, encode):
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
            if self.is_training is not None:
                decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size,
                                                                     dropout_keep_prob=1-self.drop_out)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=self.hidden_size)
            memory, state = tf.nn.dynamic_rnn(decoder_cell, self.input_data, dtype=tf.float32)
            with tf.variable_scope('temporal_attention'):
                output, alpha = self._single_attention(memory, state.cell_state.h)
            output = tf.contrib.layers.batch_norm(output)
            if self.is_training is not None:
                output = tf.layers.dropout(output, rate=self.drop_out)
            if self.classifregress == 'regress':
                logit = tf.layers.dense(output, self.num_class)
            else:
                logit = tf.layers.dense(output, self.num_class, activation='sigmoid')
        return logit

    @staticmethod
    def _multi_head_transform(v, num_head, dim):
        with tf.name_scope('multi_head_single_attention'):
            # linear projection
            with tf.variable_scope('linear_projection'):
                vp = tf.layers.dense(v, dim, use_bias=False)
            # split_heads
            with tf.variable_scope('split_head'):
                def split_last_dimension_then_transpose(tensor, num_head, dim):
                    t_shape = tensor.get_shape().as_list()
                    tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_head, dim])
                    return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, seq_len, dim]

                vs_q = split_last_dimension_then_transpose(vp, num_head, dim // num_head)

            multi_heads_output = tf.reduce_max(vs_q, axis=-2)
        return multi_heads_output

    @staticmethod
    def _single_attention(v, k):
        seq_len = v.get_shape().as_list()[-2]
        hidden_dim = k.get_shape().as_list()[-1]
        v_pro = tf.layers.dense(v, hidden_dim, use_bias=False, activation=tf.nn.tanh)
        k_ext = tf.tile(k, [1, seq_len])  # [batch_size, seq_len * hidden_dim]
        k_ext = tf.reshape(k_ext, [-1, seq_len, hidden_dim])
        att = tf.reduce_sum(tf.multiply(v_pro, k_ext), axis=-1)

        att = tf.nn.softmax(att)    # [batch_size, seq_len]
        tf.summary.histogram('aspect_attention', att)
        attention_output = tf.reduce_sum(v * tf.expand_dims(att, -1), -2)  # [batch_size, dim]
        return attention_output, att

    @staticmethod
    def _pooling(v):
        return tf.reduce_max(v, axis=-3)

    @staticmethod
    def mse_loss(logits, targets):
        return tf.losses.mean_squared_error(labels=targets, predictions=logits)

    @staticmethod
    def cross_entropy(logits, targets):
        targets = tf.cast(targets, tf.int32)
        return tf.losses.sparse_softmax_cross_entropy(labels=targets, logits=logits)


def run_epoch(session, merged, model, dataset, flag, output_log):
    total_costs = []
    state = session.run(model.initial_state)
    predictions = []
    ys = []
    iters = 0
    for x, y, news in dataset.gen_batch(flag, model.batch_size, model.num_steps, model.max_num_news, model.max_num_words):
        fetch = [model.loss, model.final_state, merged, model.logit]
        feed_dict = {model.input_data: x, model.targets: y, model.news_ph: news, model.initial_state: state}
        if flag == 'train':
            feed_dict[model.is_training] = flag
            fetch.append(model.train_op)
        else:
            fetch.append(tf.no_op())

        cost, state, summary, prediction, _ = session.run(fetch, feed_dict)
        total_costs.append(cost)
        iters += model.num_steps
        if output_log and iters % 1000 == 0:
            logger.info(f"cost {cost}, {eval_res(y, prediction, model.classifregress)}")
            sum_path = f'tensorboard/{flag}/%smodel_batch%d_h%d_d%.2f_step%d_news%d_words%d_lr%.5f'\
                       % (dataset_flag, model.batch_size, model.num_head, model.drop_out, model.num_steps,
                          model.max_num_news, model.max_num_words, model.lr)
            writer = tf.summary.FileWriter(sum_path, session.graph)
            writer.add_summary(summary, iters)
        predictions.append(prediction)
        ys.append(y)
    return np.mean(total_costs), eval_res(ys, predictions, model.classifregress)


def tuning_parameter(flag):
    config = {'num_epoch': 15, 'linear_dim': 64, 'hidden_size': 16,
              'predict_steps': 1, 'drop_out': 0.3, 'att_lambda': 0.0}
    if flag == 'tweets':
        config.update({'feature_num': 4, 'embed_size': 50})
    else:
        config.update({'feature_num': 9, 'embed_size': 300})
    for num_steps in [10, 5, 20]:
        for param_lambda in [0.001]:
            for num_head in [4, 2]:
                for lr in [0.001]:
                    for max_num_words in [10, 20, 30]:
                        for max_num_news in [10, 30, 50]:
                            for batch_size in [32, 64]:
                                config.update({'batch_size': batch_size, 'num_steps': num_steps,
                                               'param_lambda': param_lambda, 'lr': lr, "max_num_words": max_num_words,
                                               'max_num_news': max_num_news, 'num_head': num_head})
                                yield config


def main(_):

    dataset = reader.Dataset(dataset_flag)
    word_table_init, vocab_size = dataset.init_word_table()
    parameter_gen = tuning_parameter(dataset_flag)
    while True:
        try:
            config = next(parameter_gen)
            config['vocab_size'] = vocab_size
            config['classifregress'] = "regress"
        except StopIteration:
            break
        initializer = tf.contrib.layers.xavier_initializer()
        tf.reset_default_graph()
        with tf.name_scope("Train"):
            with tf.variable_scope("StockMovementPrediction", reuse=None, initializer=initializer):
                model = StockMovementPrediction(config)
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        valid_acc = 0
        flag_cnt = 0
        with tf.Session(config=gpu_config) as session:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            session.run(init, feed_dict={model.word_table_init: word_table_init})
            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters += dim.value
                total_parameters += variable_parameters
            logger.info("total parameters: %d", total_parameters)
            for i in range(config['num_epoch']):
                train_cost, eval = run_epoch(session, merged, model, dataset, 'train', True)
                logger.info(f"Epoch {i+1} train_cost {train_cost}, {eval}")
                valid_cost, eval = run_epoch(session, merged, model, dataset, 'valid', False)
                logger.info(f"Epoch {i+1} valid_cost {valid_cost}, {eval}")

                if eval['acc'] > valid_acc:
                    saver.save(session, 'model_saver/%smodel_batch%d_h%d_d%.2f_step%d_news%d_words%d_lr%.5f.ckpt' %
                               (dataset_flag, config['batch_size'], config['num_head'], config['drop_out'],
                                config['num_steps'], config['max_num_news'], config['max_num_words'], config['lr']))
                    valid_acc = eval['acc']
                    flag_cnt = 0
                else:
                    if flag_cnt > 1:
                        break
                    flag_cnt += 1
        with tf.Session(config=gpu_config) as session:
            saver.restore(session, 'model_saver/%smodel_batch%d_h%d_d%.2f_step%d_news%d_words%d_lr%.5f.ckpt' %
                          (dataset_flag, config['batch_size'], config['num_head'], config['drop_out'],
                           config['num_steps'], config['max_num_news'], config['max_num_words'], config['lr']))
            test_cost, eval = run_epoch(session, merged, model, dataset, 'test', False)
            logger.info(f"test_cost {test_cost}, {eval}")


if __name__ == "__main__":
    tf.app.run()
