import tensorflow as tf
from sklearn.metrics import matthews_corrcoef, accuracy_score,\
    roc_auc_score, mean_absolute_error, mean_squared_error
import numpy as np

_NEG_INF = -1e9


def get_padding(x, padding_value=0):
    """Return float tensor representing the padding values in x.

    Args:
      x: int tensor with any shape
      padding_value: int value that

    Returns:
      flaot tensor with same shape as x containing values 0 or 1.
        0 -> non-padding, 1 -> padding
    """
    with tf.name_scope("padding"):
        return tf.to_float(tf.equal(x, padding_value))


def get_padding_bias(x):
    """Calculate bias tensor from padding values in tensor.

    Bias tensor that is added to the pre-softmax multi-headed attention logits,
    which has shape [batch_size, num_heads, length, length]. The tensor is zero at
    non-padding locations, and -1e9 (negative infinity) at padding locations.

    Args:
      x: int tensor with shape [batch_size, length]

    Returns:
      Attention bias tensor of shape [batch_size, 1, 1, length].
    """
    with tf.name_scope("attention_bias"):
        padding = get_padding(x)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias


def eval_res(true_rate, pred_rate):
    true_rate = np.array(true_rate).reshape(-1)
    pred_rate = np.array(pred_rate).reshape(-1)
    mse = mean_squared_error(true_rate, pred_rate)
    mae = mean_absolute_error(true_rate, pred_rate)
    true_label = [1 if rate > 0 else 0 for rate in true_rate]
    pred_label = [1 if rate > 0 else 0 for rate in pred_rate]
    mcc = matthews_corrcoef(true_label, pred_label)
    acc = accuracy_score(true_label, pred_label)
    try:
        auc = roc_auc_score(true_label, pred_label)
    except:
        auc = np.float("NaN")

    return {'mcc': mcc, 'acc': acc, 'auc': auc, 'mse': mse, 'mae': mae}


def eval_res_classif(true_rate, pred_rate):
    true_rate = np.array(true_rate).reshape(-1)
    pred_rate = np.array(pred_rate).reshape(-1)
    mse = mean_squared_error(true_rate, pred_rate)
    mae = mean_absolute_error(true_rate, pred_rate)
    true_label = [1 if rate > 0.5 else 0 for rate in true_rate]
    pred_label = [1 if rate > 0.5 else 0 for rate in pred_rate]
    mcc = matthews_corrcoef(true_label, pred_label)
    acc = accuracy_score(true_label, pred_label)
    auc = roc_auc_score(true_label, pred_label)

    return {'mcc': mcc, 'acc': acc, 'auc': auc, 'mse': mse, 'mae': mae}
