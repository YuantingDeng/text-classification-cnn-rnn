import keras

from keras.models import Model, Sequential
from keras.layers import Embedding, Conv1D, Dropout, Dense, Flatten, ReLU
from keras.layers import Conv1D, SeparableConv1D
from keras.layers import AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Bidirectional, LSTM, GRU
from keras.layers import CuDNNLSTM, CuDNNGRU, SimpleRNN
from keras.optimizers import Adam

#from attention import Attention

def crnnyuan_model(num_classes, input_length=None, vocab_size=5000):
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=input_length))
    model.add(Conv1D(filters=256, kernel_size=5, activation="relu"))
    model.add(MaxPooling1D(pool_size=20))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=256, activation="relu"))#全连接，使用relu激活
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))
    return model

def cnn_model(num_classes, input_length=None, vocab_size=5000):
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=input_length))
    model.add(Conv1D(filters=256, kernel_size=5))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))
    return model
    
def multi_cnn_model(num_classes, input_length=None, vocab_size=5000):
    input = keras.layers.Input(shape=(input_length, ))
    x = Embedding(vocab_size, 32)(input)
    
    f = []
    for k in [2, 3, 5]:
        t = Conv1D(filters=64, kernel_size=k, padding="same")(x)
        f.append(GlobalMaxPooling1D()(t))
        f.append(GlobalAveragePooling1D()(t))

    f = keras.layers.Concatenate()(f)
    f = Dense(256, activation="relu")(f)
    f = Dropout(0.5)(f)
    output = Dense(units=num_classes, activation="softmax")(f)
    
    return Model(input, output)
    
def rnn_model(num_classes, input_length=None, vocab_size=5000):
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=input_length))
    model.add(Bidirectional(CuDNNLSTM(128)))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))
    return model

def crnn_model(num_classes, input_length=None, vocab_size=5000):
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=input_length))
    model.add(Conv1D(filters=256, kernel_size=5, activation="relu"))
    model.add(MaxPooling1D(pool_size=20))
    model.add(Bidirectional(CuDNNLSTM(128)))
    #model.add(MaxPooling1D(pool_size=20))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))
    return model
    
def rcnn_model(num_classes, input_length=None, vocab_size=5000):
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=input_length))
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(Conv1D(filters=256, kernel_size=5, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))
    return model