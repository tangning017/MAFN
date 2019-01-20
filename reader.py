import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm
import json
import ast

TRAIN = 'train'
times = 1.0


class Dataset(object):
    def __init__(self, tweets):
        if tweets == 'tweets':
            self.root_path = 'data/tweet/file'
            self.vec_path = 'data/tweet'
            self.feature_col = ['Open_rate', 'High_rate', 'Low_rate', 'Close_rate']
            self.label_col = ['mv']
            self.embed_size = 50
            self.training_begin_date = 20140101
            self.training_end_date = 20150801
            self.validing_end_date = 20151001
            self.testing_end_date = 20160101
            self.info = 'tweets'

        else:
            self.root_path = 'data/news/file'
            self.vec_path = "data/news"
            self.feature_col =['turnoverRate', 'accumAdjFactor', 'chgPct', 'PE',
                          'PB', 'vwap', 'high_rate', 'low_rate', 'close_rate']
            self.label_col = ['mv']
            self.embed_size = 300
            self.training_begin_date = 20140101
            self.training_end_date = 20170101
            self.validing_end_date = 20170701
            self.testing_end_date = 20180101
            self.info = 'news'

        self.train_data = None
        self.valid_data = None
        self.test_data = None
        with open(os.path.join(self.vec_path, "vocab_vec.json"), 'r') as f:
            self.dit = json.load(f)
            self.word2id = {word: i for i, word in enumerate(self.dit.keys())}

    def init_word_table(self):
        dit = self.dit
        vocab_size = len(dit) + 2   # add one unknown symbol and one all zeros padding vector

        word_table_init = np.random.random((vocab_size, self.embed_size)) * 2 - 1  # [-1.0, 1.0]
        word_table_init[-1] = np.zeros(self.embed_size)
        word2id = {word: i for i, word in enumerate(dit.keys())}
        word2vec = {word: [float(i) for i in dit[word]] for word in dit.keys()}
        for word in word2id.keys():
            try:
                word_table_init[word2id[word]] = word2vec[word]
            except:
                print(word, len(word2vec[word]))
        return word_table_init, vocab_size

    def news_iterator(self, flag, num_step, max_news_sequence, max_word_sequence):
        if self.train_data is None:
            self.news_raw_data()
        if flag == "train":
            data = self.train_data
        elif flag == "valid":
            data = self.valid_data
        else:
            data = self.test_data
        word2id = self.word2id

        preprocess_path = f'{flag}_preprocess_{num_step}_{max_news_sequence}_{max_word_sequence}.pkl'
        preprocess_path = os.path.join(self.root_path, preprocess_path)
        x_all = []
        y_all = []
        news_all = []
        if os.path.exists(preprocess_path):
            pdata = pickle.load(open(preprocess_path, 'rb'))
            x_all = pdata["x_all"]
            y_all = pdata["y_all"]
            news_all = pdata["news_all"]
        else:
            for i in tqdm(range(len(data[0]))):
                prices = data[0][i]
                label = data[1][i]
                news_lis = list(data[2][i])
    #             print(len(news_lis), len(prices), len(label))
                for ind in range(len(prices)-num_step):
                    x = []
                    y = list(label.iloc[ind+num_step-1])  # the label of last step
                    news = []
                    for j in range(num_step):
                        x += [list(prices.iloc[ind+j])]
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
                                if self.info == 'news':
                                    word_id = [word2id[word] if word in word2id else len(word2id)
                                               for word in ast.literal_eval(news_l[k])]
                                else:
                                    word_id = [word2id[word] if word in word2id else len(word2id) for word in news_l[k]]
                                for _ in range(len(word_id), max_word_sequence):
                                    word_id.append(len(word2id)+1)
                                tmp_word_id.append(word_id[:max_word_sequence])
                        else:  # daily news number smaller than max_sequence we fill with zeros
                            for k in range(len(news_l)):
                                if self.info == 'news':
                                    word_id = [word2id[word] if word in word2id else len(word2id)
                                               for word in ast.literal_eval(news_l[k])]
                                else:
                                    word_id = [word2id[word] if word in word2id else len(word2id) for word in news_l[k]]
                                for _ in range(len(word_id), max_word_sequence):
                                    word_id.append(len(word2id)+1)
                                tmp_word_id.append(word_id[:max_word_sequence])
                            tmp_word_id.extend([np.full(max_word_sequence, len(word2id)+1)
                                                for _ in range(len(news_l), max_news_sequence)])    # padding
                        news += [tmp_word_id]
                        # print(tmp_word_id, news_l[k])
                    x_all += [x]
                    y_all += [y]
                    news_all += [news]
            preprocess_data = {"x_all": x_all, "y_all": y_all, "news_all": news_all}
            with open(preprocess_path, 'wb') as f:
                pickle.dump(preprocess_data, f)
        return np.array(x_all, dtype=np.float32),\
                np.array(y_all, dtype=np.float32)*times, \
               np.array(news_all, dtype=np.int64)

    def gen_batch(self, flag, batch_size, num_step, max_news_sequence, max_word_sequence):
        x_all, y_all, news_all = self.news_iterator(flag, num_step, max_news_sequence, max_word_sequence)

        if batch_size is not None:
            for batch in range(len(x_all)//batch_size):
                indices = list(range(len(x_all)))
                if flag == TRAIN:
                    np.random.shuffle(indices)
                x_batch = []
                y_batch = []
                news_batch = []
                for i in range(batch_size):
                    x_batch += [x_all[indices[i]]]
                    y_batch += [y_all[indices[i]]]
                    news_batch += [news_all[indices[i]]]
                # print(np.array(x_batch))
                yield np.array(x_batch, dtype=np.float32),\
                    np.array(y_batch, dtype=np.float32)*times,\
                    np.array(news_batch, dtype=np.int64)

    def rf_feature_union(self, flag):
        if self.train_data is None:
            self.news_raw_data()
        dit = self.dit
        if flag == "train":
            data = self.train_data
        elif flag == "valid":
            data = self.valid_data
        else:
            data = self.test_data
        feature_all = []
        label_all = []
        for i in tqdm(range(len(data[0]))):
            prices = data[0][i]
            label_lis = data[1][i]
            news_lis = list(data[2][i])
            for ind in range(len(prices)):
                feature_price = prices.iloc[ind].values
                label = label_lis.iloc[ind].values
                if len(str(news_lis[ind])) < 4:
                    daily_news_rep = np.zeros(self.embed_size)
                else:
                    news_l = ast.literal_eval(news_lis[ind])
                    daily_news = []
                    for k in range(len(news_l)):
                        if self.info == 'news':
                            word_vec = [dit[word] for word in ast.literal_eval(news_l[k]) if word in dit]
                        else:
                            word_vec = [dit[word] for word in news_l[k] if word in dit]
                        if len(word_vec) != 0:
                            one_news = np.array(word_vec).max(axis=0)  # max pooling between words
                            daily_news.append(one_news)
                    if len(daily_news) != 0:
                        daily_news_rep = np.mean(daily_news, axis=0)   # average pooling
                    else:
                        daily_news_rep = np.zeros(self.embed_size)
                # print(np.array(feature_price).shape, np.array(daily_news_rep).shape)
                feature_all.append(np.concatenate((feature_price, daily_news_rep)))
                # feature_all.append(feature_price)
                label_all.append(label*times)
        return feature_all, label_all

    def series_feature_union(self):
        files = os.listdir(self.root_path)
        files = [file for file in files if 'csv' in file]
        history = []
        test = []
        for file in files:
            df = pd.read_csv(os.path.join(self.root_path, file))
            df['mv'] = df['mv'].astype('float32')
            df['date'] = df['date'].astype('int64')
            if self.info == 'news':
                df = df[(df.mv < -0.0015) | (df.mv > 0.003)]
            else:
                df = df[(df.mv < -0.005) | (df.mv > 0.0055)]
            df = df.sort_values(by=['date'], ascending=True)
            # each dataframe contains one stock
            history.append(df[(df.date >= self.training_begin_date) & (df.date < self.validing_end_date)][self.label_col].values*times)
            test.append(df[(df.date >= self.validing_end_date) & (df.date < self.testing_end_date)][self.label_col].values*times)
        return history, test

    def news_raw_data(self):
        files = os.listdir(self.root_path)
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
            df = pd.read_csv(os.path.join(self.root_path, file))
            df['mv'] = df['mv'].astype('float32')
            df['date'] = df['date'].astype('int64')
            if self.info == 'news':
                df = df[(df.mv < -0.0015)|(df.mv > 0.003)]
            else:
                df = df[(df.mv < -0.005)|(df.mv > 0.0055)]
            df = df.sort_values(by=['date'], ascending=True)
            # each dataframe contains one stock
            training_prices += [df[(df.date>=self.training_begin_date)&(df.date<self.training_end_date)][self.feature_col].reset_index(drop=True)]
            training_label += [df[(df.date>=self.training_begin_date)&(df.date<self.training_end_date)][self.label_col].reset_index(drop=True)]
            training_news += [df[(df.date>=self.training_begin_date)&(df.date<self.training_end_date)][self.info]]

            validing_prices += [df[(df.date>=self.training_end_date)&(df.date<self.validing_end_date)][self.feature_col].reset_index(drop=True)]
            validing_label += [df[(df.date>=self.training_end_date)&(df.date<self.validing_end_date)][self.label_col].reset_index(drop=True)]
            validing_news += [df[(df.date>=self.training_end_date)&(df.date<self.validing_end_date)][self.info]]

            testing_prices += [df[(df.date>=self.validing_end_date)&(df.date<self.testing_end_date)][self.feature_col].reset_index(drop=True)]
            testing_label += [df[(df.date>=self.validing_end_date)&(df.date<self.testing_end_date)][self.label_col].reset_index(drop=True)]
            testing_news += [df[(df.date>=self.validing_end_date)&(df.date<self.testing_end_date)][self.info]]

        self.train_data = (training_prices, training_label, training_news)
        self.valid_data = (validing_prices, validing_label, validing_news)
        self.test_data = (testing_prices, testing_label, testing_news)


if __name__ == "__main__":
    tweets = Dataset('tweets')
    tweets.series_feature_union()