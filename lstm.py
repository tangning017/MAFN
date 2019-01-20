from keras.layers import LSTM, Dense, InputLayer, BatchNormalization, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import reader_v1 as reader
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from utils import eval_res

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)


def act(x):
    return 2*(tf.nn.sigmoid(x)-0.5)


def create_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(10, 4)))
    model.add(BatchNormalization())
    model.add(LSTM(32))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='Adam', loss='mse', metrics=['mae', 'mape'])
    return model


def run_epoch():
    train_data, valid_data, test_data = reader.news_raw_data()
    model = create_model()
    x, y = reader.news_iterator_v1(train_data, 32, 10, 10, 10, 'train')
    valid_x, valid_y = reader.news_iterator_v1(valid_data, 32, 10, 10, 10, 'valid')
    early_stop = EarlyStopping(patience=1)
    model.fit(x, y, epochs=10, batch_size=32, verbose=1, validation_data=(valid_x, valid_y), callbacks=[early_stop])
    x, y = reader.news_iterator_v1(test_data, 32, 10, 10, 10, 'test')
    preds = model.predict(x)
    print(f"test {len(y)} res: ", eval_res(preds, y))
    print(preds.reshape(-1))
    print(y.reshape(-1))
    model.save("weights/my_model.h5")


if __name__ == "__main__":
    run_epoch()
