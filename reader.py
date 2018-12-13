import numpy as np
# import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import pickle
from tqdm import tqdm
import json
import ast
#root_path = 'data/news'
#preprocess_path = os.path.join(root_path, "file/preprocess.pkl")
#feature_col =['actPreClosePrice','openPrice','highestPrice','lowestPrice',\
#              'closePrice','turnoverRate','accumAdjFactor','chgPct','PE','PE1','PB','vwap']
#embed_size = 300
#training_end_date = 20170101
#validing_end_date = 20170701
#info = 'news'

root_path = 'data/tweet'
feature_col =['high', 'low', 'close']
label_col = ['label', 'label3', 'label5', 'label10', 'label20']
embed_size = 50
training_begin_date = 20140101
training_end_date = 20150801
validing_end_date = 20151001
testing_end_date = 20160101
info = 'tweets'

TEST = 'NONE'

class Preprocess_data(object):
    def __init__(self, x_all, y_all, y_step_all, news_all, stock_id):
        self.x_all = x_all
        self.y_all = y_all
        self.y_step_all = y_step_all
        self.news_all = news_all
        self.stock_id = stock_id

def news_iterator(data, batch_size, num_step, max_sequence, flag):
    with open(os.path.join(root_path, "vocab_vec.json"), 'r') as f:
        dit = json.load(f)
    preprocess_path = 'file2/train_preprocess.pkl'
    if flag == 'valid':
        preprocess_path = 'file2/valid_preprocess.pkl' 
    if flag == 'test':
        preprocess_path = 'file2/test_preprocess.pkl'
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
#             print(prices)
            label = data[1][i]
            news_lis = list(data[2][i])
            print(len(news_lis), len(prices), len(label))
            if len(prices) <= num_step:
                continue
            for steps in range(len(prices)//num_step):# divid each stock data into steps piece with num_step
                if flag != TEST:
                    ind = np.random.randint(0, len(prices)-num_step)
                else:  ## 顺序
                    ind = num_step * steps
                x = []
                y = list(label.iloc[ind+num_step-1]) ## the label of last step
                y_step = []
                news = []
                for j in range(num_step):
                    x += [list(prices.iloc[ind+j])]
#                     y += [list(label.iloc[ind+j])]
                    y_step += [list(label.iloc[ind+j])[0]]
                    tmp_vec = []
                    for k in range(max_sequence):
                        if len(str(news_lis[ind+j])) < 4:
                            tmp_vec += [np.zeros(embed_size)]
                            continue
                        news_l = ast.literal_eval(news_lis[ind+j])
                        if k < len(news_l):
                            if len(news_l) > max_sequence:
                                m = np.random.randint(0, len(news_l))
                                vec = np.zeros(embed_size)
#                                 print(news_l)
                                lis = news_l[m]
                                for word in lis:
                                    if word in dit:
                                        vec = np.add(vec, dit[word])
                                tmp_vec += [vec/len(vec)]
                            else:
                                vec = np.zeros(embed_size)
                                lis = news_l[k]
                                for word in lis:
                                    if word in dit:
                                        vec = np.add(vec, dit[word])
                                tmp_vec += [vec/len(vec)]
                        else:
                            tmp_vec += [np.zeros(embed_size)]
                    news += [tmp_vec]
                x_all += [x]
                y_all += [y]
                y_step_all += [y_step]
                news_all += [news]
                stock_id += [i]
        pdata = Preprocess_data(x_all, y_all, y_step_all, news_all, stock_id)
        pickle.dump(pdata, open(preprocess_path, 'wb'))
    for batch in range(len(x_all)//batch_size):
        indices = list(range(len(x_all)))
        if flag != TEST:
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
        yield np.array(x_batch, dtype=np.float32), np.array(y_batch, dtype=np.int64),\
                np.array(y_step_batch, dtype=np.int64),np.array(news_batch, dtype=np.float32),\
                np.array(stock_id_batch, dtype=np.int64)

def news_raw_data(data_path):
    if not os.path.exists(os.path.join(root_path, "file2/train_data.pkl")):
        files = os.listdir(os.path.join(root_path, "file2/"))
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
            df = pd.read_csv(os.path.join(root_path, data_path,file))
            df['mv'] = df['mv'].apply(lambda x: float(x))
            df = df[(df.mv<-0.005)|(df.mv>0.0055)]
            df = df.sort_values(by=['date'], ascending=True).iloc[:-20]
            training_prices += [df[(df.date>training_begin_date)&(df.date<training_end_date)][feature_col].reset_index(drop=True)]
            training_label += [df[(df.date>training_begin_date)&(df.date<training_end_date)][label_col]]
            training_news += [df[(df.date>training_begin_date)&(df.date<training_end_date)][info]]
            validing_prices += [df[(df.date>training_end_date)&(df.date<validing_end_date)][feature_col].reset_index(drop=True)]
            validing_label += [df[(df.date>training_end_date)&(df.date<validing_end_date)][label_col]]
            validing_news += [df[(df.date>training_end_date)&(df.date<validing_end_date)][info]]
            testing_prices += [df[(df.date>validing_end_date)&(df.date<testing_end_date)][feature_col].reset_index(drop=True)]
            testing_label += [df[(df.date>validing_end_date)&(df.date<testing_end_date)][label_col]]
            testing_news += [df[(df.date>validing_end_date)&(df.date<testing_end_date)][info]]
        with open(os.path.join(root_path,"file2/train_data.pkl"), 'wb') as f:
            pickle.dump((training_prices, training_label, training_news), f)
        with open(os.path.join(root_path, "file2/valid_data.pkl"), 'wb') as f:
            pickle.dump((validing_prices, validing_label, validing_news), f)
        with open(os.path.join(root_path, 'file2/test_data.pkl'), 'wb') as f:
            pickle.dump((testing_prices, testing_label, testing_news), f)
    else:
        with open(os.path.join(root_path, "file2/train_data.pkl"), 'rb') as f:
            data = pickle.load(f)
            training_prices, training_label, training_news = data[0], data[1], data[2]
        with open(os.path.join(root_path, "file2/valid_data.pkl"), 'rb') as f:
            data = pickle.load(f)
            validing_prices, validing_label, validing_news = data[0], data[1], data[2]
        with open(os.path.join(root_path, 'file2/test_data.pkl'), 'rb') as f:
            data = pickle.load(f)
            testing_prices, testing_label, testing_news = data[0], data[1], data[2]

    return (training_prices, training_label, training_news),\
        (validing_prices, validing_label, validing_news),\
        (testing_prices, testing_label, testing_news)


if __name__ == "__main__":
#     if not os.path.exists('data/training_data.pkl'):
    training_data, validing_data, testing_data = news_raw_data("file2/")
#         pickle.dump(training_data, open("data/training_data.pkl"))
#         pickle.dump(validing_data, open("data/validing_data.pkl"))
#         pickle.dump(testing_data, open("data/testing_data.pkl"))
#     else:
#     print(max_sequence)
#     x, y, news = news_iterator(testing_data, 32, 10, 10)
#     print(x.shape, y.shape, news.shape)
    for a, b, c, d in news_iterator(testing_data, 32, 10, 10, 'test'):
        print(a.shape, b.shape, c.shape, d.shape)
