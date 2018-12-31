from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import reader
import os
import datetime
import logging
import logging.config
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import math
import numpy as np

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

EMBEDDING_DIM = 50
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_CLASS = 2
PREDICT_STEPS = 5
MAX_GRAD_NORM = 15
NUM_EPOCH = 50
LINEAR_DIM = 64
DECAY_STEP = 10
DECAY_RATE = 0.98
STOCK_SIZE = 87
LAMBDA = 0.01
FEATURE_NUM = 3


class StockMovementPrediction(object):
    def __init__(self, is_training, batch_size, num_steps, linear_dim, num_head, drop_out, max_num_news, lr):
        self.linear_dim = linear_dim
        self.num_head = num_head
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.is_training = is_training
        self.drop_out = drop_out
        self.max_num_news = max_num_news
        self.lr = lr
        self.att_loss = 0
        self.hidden_size = HIDDEN_SIZE

        assert self.linear_dim % self.num_head == 0

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(tf.float32, [None, self.num_steps, FEATURE_NUM], name="price")
            self.targets = tf.placeholder(tf.int64, [None, PREDICT_STEPS], name='label')
            self.step_targets = tf.placeholder(tf.int64, [None, self.num_steps], name='step_label')
            self.news = tf.placeholder(tf.float32, [None, self.num_steps, self.max_num_news, EMBEDDING_DIM],
                                       name='news')
            self.stock_id = tf.placeholder(tf.int64, [None], name='stocks')

        logger.info(
            f"embedding_size:{EMBEDDING_DIM}, max_num_news:{self.max_num_news},lr:{self.lr}, batch_size:{self.batch_size}, num_head:{self.num_head}, drop_out:{self.drop_out}, num_step:{self.num_steps}")

        outputs = self.encode()
        logits = self.decode(outputs, self.final_state)
        with tf.name_scope("loss_function"):
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.targets, logits=logits)
            self.prediction = tf.argmax(logits, -1)
            result = tf.cast(tf.equal(self.prediction, self.targets), tf.float32)
            self.acc = [tf.reduce_mean(result[:, ind]) for ind in range(PREDICT_STEPS)]
            self.acc += [tf.reduce_mean(result)]

        if not self.is_training:
            return

        ################### Optimizer #############################
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(self.lr, global_step, DECAY_STEP, DECAY_RATE, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

    def encode(self):
        lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size)
        if self.is_training:
            lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size, dropout_keep_prob=1 - self.drop_out)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN_encoder"):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                with tf.name_scope('multihead_attention'):
                    input1 = self._multi_head_self_attention(self.news[:, time_step, :, :])
                    input1 = tf.concat([self.input_data[:, time_step, :], input1], -1)
                cell_output, state = cell(input1, state)
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
        with tf.variable_scope("RNN_decoder"):
            ### [batch_size, max_time, num_units]
            attention_states = tf.transpose(encode, [1, 0, 2])
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.hidden_size, attention_states)

            decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size)
            if self.is_training:
                decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size,
                                                                     dropout_keep_prob=1 - self.drop_out)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=self.hidden_size)

            helper = tf.contrib.seq2seq.TrainingHelper(outputs, [PREDICT_STEPS for _ in range(self.batch_size)],
                                                       time_major=True)
            decoder_initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=state[0])
            projection_layer = layers_core.Dense(units=NUM_CLASS, use_bias=False)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                                      output_layer=projection_layer)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = outputs.rnn_output
        return logits

    def _multi_head_self_attention(self, v, num_head, max_seq_len, dim, drop_out):
        with tf.name_scope('transformer'):
            # linear projection
            with tf.variable_scope('linear_projection'):
                v = tf.tile(v, [1, 1, num_head])
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
            #             depth = dim // num_head
            #             vs_q *= depth ** -0.5

            #             bias = tf.get_variable("self_attention_bias")
            # scaled_dot_product
            with tf.variable_scope('scaled_dot_product'):
                logits = tf.matmul(vs_q, vs_k, transpose_b=True)
                #                 logits += bias
                weights = tf.nn.softmax(logits, name="attention_softmax")
                if self.is_training:
                    weights = tf.nn.dropout(weights, 1.0 - drop_out)
                attention_output = tf.matmul(weights, vs_v)  # [batch_size, num_head, max_sequence_len, dim]

        return attention_output


def run_epoch(session, merged, model, data, train_op, flag, output_log):
    total_costs = 0.0
    iters = 0
    cnt = 0
    total_auc = 0
    all_acc = np.zeros(PREDICT_STEPS + 1)
    all_tn = all_tp = all_fp = all_fn = 0
    state = session.run(model.initial_state)
    for x, y, y_step, news, stockid in reader.news_iterator(data, model.batch_size, model.num_steps, model.max_num_news,
                                                            flag):
        cost, acc, summary, state, _, prediction = session.run(
            [model.loss, model.acc, merged, model.final_state, train_op, model.prediction], {model.input_data: x,
                                                                                             model.targets: y,
                                                                                             model.step_targets: y_step,
                                                                                             model.news: news,
                                                                                             model.initial_state: state,
                                                                                             model.stock_id: stockid})
        cnt += 1
        total_costs += cost
        total_auc += roc_auc_score(y.reshape(-1), prediction.reshape(-1))
        iters += model.num_steps
        for i in range(PREDICT_STEPS + 1):
            all_acc[i] += acc[i]
        tn, fp, fn, tp = confusion_matrix(y_true=y.reshape(-1), y_pred=prediction.reshape(-1)).ravel()
        all_tn += tn
        all_fp += fp
        all_fn += fn
        all_tp += tp
        mcc = (all_tp * all_tn - all_fp * all_fn) / math.sqrt(
            (all_tp + all_fp) * (all_tp + all_fn) * (all_tn + all_fp) * (all_tn + all_fn))
        if output_log and iters % 100 == 0:
            logger.info("After %d steps, cost is %.5f acc %.5f auc %.5f mcc %.5f" % (
            iters, cost, all_acc[0] / cnt, total_auc / cnt, mcc))

    return total_costs / cnt, all_acc / cnt, summary, total_auc / cnt


# def tuning_parameter():
#     for batch_size in [4, 8, 16, 32]:
#         for num_head in [3, 5, 8]:
#             for max_num_news in [30, 20, 10]:
#                 for lr in [0.1, 0.01, 0.001]:
#                     for drop_out in [0.3, 0.5, 0.1]:
#                         for num_step in [10, 15, 20, 5]:
#                             yield max_num_news, lr, batch_size, num_head, drop_out, num_step

def main(_):
    train_data, valid_data, test_data = reader.news_raw_data()
    #     parameter_gen = tuning_parameter()
    max_num_news, lr, batch_size, num_head, drop_out, num_steps = 10, 0.01, 32, 4, 0.0, 10
    #     while True:
    #         try:
    #             max_num_news, lr, batch_size, num_head, drop_out, num_steps = next(parameter_gen)
    #         except:
    #             break
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
    #     tf.reset_default_graph()
    #     if os.path.exists(os.path.join(root_path, DATA_PATH, f"{info}train_preprocess.pkl")):
    #         os.remove(os.path.join(root_path, DATA_PATH, f'{info}train_preprocess.pkl'))
    #         os.remove(os.path.join(root_path, DATA_PATH, f'{info}valid_preprocess.pkl'))
    #         os.remove(os.path.join(root_path, DATA_PATH, f'{info}test_preprocess.pkl'))
    with tf.name_scope("Train"):
        with tf.variable_scope("StockMovementPrediction", reuse=None, initializer=initializer):
            train_model = StockMovementPrediction(True, batch_size, num_steps, LINEAR_DIM, num_head, drop_out,
                                                  max_num_news, lr)
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("tensorboard/", tf.Session().graph)
    merged = tf.summary.merge_all()
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        session.run(tf.initializers.global_variables())
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
            train_cost, acc, summary, auc = run_epoch(session, merged, train_model, train_data, train_model.train_op,
                                                      'train', True)
            logger.info("Epoch: %d Training average Cost: %.5f auc is %.5f" % (i + 1, train_cost, auc))
            writer.add_summary(summary, i)
            valid_cost, acc, _, auc = run_epoch(session, merged, train_model, valid_data, tf.no_op(), 'valid', False)
            logger.info("Epoch: %d Validation Cost: %.5f, auc is %.5f" % (i + 1, valid_cost, auc))
            for i in range(PREDICT_STEPS + 1):
                logger.info("predict step %d acc: %.5f", i, acc[i])
        test_cost, acc, _, auc = run_epoch(session, merged, train_model, test_data, tf.no_op(), 'test', False)
        logger.info("Test Cost: %.3f, auc is %.5f" % (test_cost, auc))
        for i in range(PREDICT_STEPS + 1):
            logger.info("predict step %d acc: %.5f", i, acc[i])
        saver.save(session,
                   f'model_saver/{info}model_btch{batch_size}_h{num_head}_d{drop_out}_step{num_steps}_news{max_num_news}_lr{lr}.ckpt')


if __name__ == "__main__":
    tf.app.run()
