import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm
import json
import ast


# root_path = 'data/news/file'
# vec_path = "data/news"
# feature_col =['turnoverRate', 'accumAdjFactor', 'chgPct', 'PE',
#               'PB', 'vwap', 'high_rate', 'low_rate', 'close_rate']
# # feature_col = ['openPrice', 'highestPrice', 'lowestPrice', 'closePrice']
# # label_col = ['label', 'label2', 'label3', 'label4', 'label5']
# label_col = ['mv', 'mv2', 'mv3', 'mv4', 'mv5']
# embed_size = 300
# training_begin_date = 20140101
# training_end_date = 20170101
# validing_end_date = 20170701
# testing_end_date = 20180301
# info = 'news'

root_path = 'data/tweet/file'
vec_path = 'data/tweet'
feature_col = ['Open_rate', 'High_rate', 'Low_rate', 'Close_rate']
# label_col = ['label', 'label2', 'label3', 'label4', 'label5']
label_col = ['mv', 'mv2', 'mv3', 'mv4', 'mv5']
embed_size = 50
training_begin_date = 20140101
training_end_date = 20150801
validing_end_date = 20151001
testing_end_date = 20160101
info = 'tweets'
# #
TRAIN = 'train'


class Preprocess_data(object):
    def __init__(self, x_all, y_all, y_step_all, news_all, stock_id):
        self.x_all = x_all
        self.y_all = y_all
        self.y_step_all = y_step_all
        self.news_all = news_all
        self.stock_id = stock_id


def init_word_table():
    with open(os.path.join(vec_path, "vocab_vec.json"), 'r') as f:
        dit = json.load(f)
        vocab_size = len(dit) + 2   # add one unknown symbol and one all zeros padding vector

    word_table_init = np.random.random((vocab_size, embed_size)) * 2 - 1  # [-1.0, 1.0]
    word2id = {word: i for i, word in enumerate(dit.keys())}
    word2vec = {word: [float(i) for i in dit[word]] for word in dit.keys()}
    for word in word2id.keys():
        try:
            word_table_init[word2id[word]] = word2vec[word]
        except:
            print(word, len(word2vec[word]))
    return word_table_init, vocab_size


def news_iterator(data, batch_size, num_step, max_news_sequence, max_word_sequence, flag):
    with open(os.path.join(vec_path, "vocab_vec.json"), 'r') as f:
        dit = json.load(f)
        word2id = {word: i for i, word in enumerate(dit.keys())}

    preprocess_path = 'train_preprocess.pkl'
    if flag == 'valid':
        preprocess_path = 'valid_preprocess.pkl' 
    if flag == 'test':
        preprocess_path = 'test_preprocess.pkl'
    preprocess_path = os.path.join(root_path, preprocess_path)
    x_all = []
    y_all = []
    y_step_all = []
    news_all = []
    stock_id = []
    if os.path.exists(preprocess_path):
        pdata = pickle.load(open(preprocess_path, 'rb'))
        x_all = pdata.x_all
        y_all = pdata.y_all
        y_step_all = pdata.y_step_all
        news_all = pdata.news_all
        stock_id = pdata.stock_id
    else:
        for i in tqdm(range(len(data[0]))):
            prices = data[0][i]
            label = data[1][i]
            news_lis = list(data[2][i])
#             print(len(news_lis), len(prices), len(label))
            for ind in range(len(prices)-num_step): 
                x = []
                y = list(label.iloc[ind+num_step-1])  # the label of last step
                y_step = []
                news = []
                for j in range(num_step):
                    x += [list(prices.iloc[ind+j])]
                    y_step += [list(label.iloc[ind+j])[0]]  # label1
                    tmp_word_id = []
                    if len(str(news_lis[ind+j])) < 4:
                        news.append(np.full((max_news_sequence, max_word_sequence), len(word2id)+1))  # padding
                        continue
                    else:
                        news_l = ast.literal_eval(news_lis[ind+j])
                    if len(news_l) > max_news_sequence:  # daily news number larger than max_sequence we sample
                        indices = list(range(len(news_l)))
                        np.random.shuffle(indices)
                        for k in indices[:max_news_sequence]:
                            if info == 'news':
                                word_id = [word2id[word] if word in word2id else len(word2id)
                                           for word in ast.literal_eval(news_l[k])]
                            else:
                                word_id = [word2id[word] if word in word2id else len(word2id) for word in news_l[k]]
                            for _ in range(len(word_id), max_word_sequence):
                                word_id.append(len(word2id)+1)
                            tmp_word_id.append(word_id[:max_word_sequence])
                    else:  # daily news number smaller than max_sequence we fill with zeros
                        for k in range(len(news_l)):
                            if info == 'news':
                                word_id = [word2id[word] if word in word2id else len(word2id)
                                           for word in ast.literal_eval(news_l[k])]
                            else:
                                word_id = [word2id[word] if word in word2id else len(word2id) for word in news_l[k]]
                            for _ in range(len(word_id), max_word_sequence):
                                word_id.append(len(word2id)+1)
                            tmp_word_id.append(word_id[:max_word_sequence])
                        tmp_word_id.extend([np.full(max_word_sequence, len(word2id)+1)
                                            for _ in range(len(news_l), max_news_sequence)])
                    news += [tmp_word_id]
                    # print(tmp_word_id, news_l[k])
                x_all += [x]
                y_all += [y]
                y_step_all += [y_step]
                news_all += [news]
                stock_id += [i]
        preprocess_data = Preprocess_data(x_all, y_all, y_step_all, news_all, stock_id)
        pickle.dump(preprocess_data, open(preprocess_path, 'wb'))

    for batch in range(len(x_all)//batch_size):
        indices = list(range(len(x_all)))
        if flag == TRAIN:
            np.random.shuffle(indices)
        x_batch = []
        y_batch = []
        y_step_batch = []
        news_batch = []
        stock_id_batch = []
        for i in range(batch_size):
            x_batch += [x_all[indices[i]]]
            y_batch += [y_all[indices[i]]]
            y_step_batch += [y_step_all[indices[i]]]
            news_batch += [news_all[indices[i]]]
            stock_id_batch += [stock_id[indices[i]]]
#         print(news_batch)
        yield np.array(x_batch, dtype=np.float32),\
            np.array(y_batch, dtype=np.float32),\
            np.array(y_step_batch, dtype=np.int64),\
            np.array(news_batch, dtype=np.int64),\
            np.array(stock_id_batch, dtype=np.int64)


def classif_feature_union(data):
    with open(os.path.join(vec_path, "vocab_vec.json"), 'r') as f:
        dit = json.load(f)

    feature_all = []
    label_all = []

    for i in tqdm(range(len(data[0]))):
        prices = data[0][i]
        label = data[1][i]
        news_lis = list(data[2][i])

        for ind in range(len(prices)):
            feature_price = prices.iloc[ind].values
            label = label[ind]
            daily_news_rep = []
            if len(str(news_lis[ind])) < 4:
                daily_news_rep.append(np.zeros(embed_size))
            else:
                news_l = ast.literal_eval(news_lis[ind])
            daily_news = []
            for k in range(len(news_l)):
                if info == 'news':
                    word_vec = [dit[word] for word in ast.literal_eval(news_l[k]) if word in dit]
                else:
                    word_vec = [dit[word] for word in news_l[k] if word in dit]
                one_news = np.array(word_vec).max(axis=-2)  # max pooling between words
                daily_news.append(one_news)
            if len(daily_news) != 0:
                daily_news_rep = np.mean(daily_news, axis=-2)   # average pooling
            feature_all.append(np.concatenate((feature_price, daily_news_rep), axis=0))
            label_all.append(label)
    return feature_all, label_all


def news_raw_data():
    files = os.listdir(root_path)
    files = [file for file in files if 'csv' in file]
    training_prices = []
    training_label = []
    training_news = []
    validing_prices = []
    validing_label = []
    validing_news = []
    testing_prices = []
    testing_label = []
    testing_news = []

    for file in files:
        df = pd.read_csv(os.path.join(root_path,file))
        df['mv'] = df['mv'].astype('float32')
        df['date'] = df['date'].astype('int64')
        df = df[(df.mv < -0.005)|(df.mv > 0.0055)]
        df = df.sort_values(by=['date'], ascending=True)
        # each dataframe contains one stock
        training_prices += [df[(df.date>=training_begin_date)&(df.date<training_end_date)][feature_col].reset_index(drop=True)]
        training_label += [df[(df.date>=training_begin_date)&(df.date<training_end_date)][label_col].reset_index(drop=True)]
        training_news += [df[(df.date>=training_begin_date)&(df.date<training_end_date)][info]]

        validing_prices += [df[(df.date>=training_end_date)&(df.date<validing_end_date)][feature_col].reset_index(drop=True)]
        validing_label += [df[(df.date>=training_end_date)&(df.date<validing_end_date)][label_col].reset_index(drop=True)]
        validing_news += [df[(df.date>=training_end_date)&(df.date<validing_end_date)][info]]

        testing_prices += [df[(df.date>=validing_end_date)&(df.date<testing_end_date)][feature_col].reset_index(drop=True)]
        testing_label += [df[(df.date>=validing_end_date)&(df.date<testing_end_date)][label_col].reset_index(drop=True)]
        testing_news += [df[(df.date>=validing_end_date)&(df.date<testing_end_date)][info]]

    return (training_prices, training_label, training_news),\
        (validing_prices, validing_label, validing_news),\
        (testing_prices, testing_label, testing_news)


if __name__ == "__main__":
    training_data, validing_data, testing_data = news_raw_data()
    print(len(training_data[0]))
    for a, b, c, d, f in news_iterator(training_data, 32, 10, 10, 10, 'train'):
        print(a.shape, b.shape, c.shape, d.shape)
