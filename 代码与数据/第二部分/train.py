import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from models import *
from data.cnews_loader import build_vocab
import datetime
import time
import random

fw = open('1012.txt', 'a+', encoding='utf-8')


length = 500
# categories = ["中医内科", "中医儿科", "中医外科", "中医妇科", "针灸推拿", "中医养生"]
categories = ['妈咪爱', '思密达', '蒙脱石', '头孢', '补钙', '健脾散', '益生菌', '醒脾养儿颗粒', '鱼肝油', '钙剂'\
              ,'葡萄糖酸','维生素D','阿莫西林','玉屏风','参苓白术散','利巴韦林','六味地黄','虚汗停','鸡内金','健脾丸']


def load_data(file):
    xs, ys = [], []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            category, content = line.split("\t")
            content = content.strip()

            seq = np.zeros((length,), dtype=np.int64)
            for i in range(min(len(content), length)):
                if (content[i] in vocab):
                    seq[i] = vocab[content[i]]
            xs.append(seq)

            # c = categories.index(category)
            # ys.append(keras.utils.to_categorical(c, len(categories)))

            words = category.strip().split(' ')
            c = np.zeros(len(categories))
            for word in words:
                c[categories.index(word)] = 1
            ys.append(c)

    return np.stack(xs), np.stack(ys)

def batch_iter(data, batch_size=64):
    """生成批次数据"""

    l = []

    for i in range(len(data[0])):
        t = []
        t.append(data[0][i])
        t.append(data[1][i])
        l.append(t)

    while True:
        temp = []
        x, y = [], []
        temp.extend(random.sample(l, batch_size))
        for each in temp:
            x.append(each[1])
            y.append(each[0])
        x_shuffle = np.array(x)
        y_shuffle = np.array(y)
        indices = np.random.permutation(np.arange(len(x)))
        x_shuffle = x_shuffle[indices]
        y_shuffle = y_shuffle[indices]

        yield x_shuffle, y_shuffle

def train():
    model = cnn_model(num_classes=len(categories), input_length=length)
    # model = multi_cnn_model(num_classes=len(categories), input_length=length)
    # model = rnn_model(num_classes=len(categories), input_length=length)
    #model = crnn_model(num_classes=len(categories), input_length=length)
    # model = rcnn_model(num_classes=len(categories), input_length=length)
    model.summary()
    model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
    )

    # model.fit(
    # train_xs,
    # train_ys,
    # validation_data=(val_xs, val_ys),
    # shuffle=True,
    # epochs=5,
    # batch_size=64
    # )
    model.fit_generator(batch_iter(train_data), steps_per_epoch=400, validation_data=(val_xs, val_ys), epochs=5)
    model_dir='model/'+modelname+'.h5'
    model.save(model_dir)

    # test_ys = [t.argmax() for t in test_ys]
    # pred = model.predict(test_xs).argmax(-1)
    pred = model.predict(test_xs)


    for index in range(len(yuzhi)):
        accuracy = 0.0
        precision = 0
        recall = 0
        for i in range(len(pred)):
            bing = 0
            jiao = 0
            yuce = 0
            yuanlai = 0
            for j in range(len(pred[i])):
                if pred[i][j] > yuzhi[index] and test_ys[i][j] == 1:
                    jiao += 1
                if pred[i][j] > yuzhi[index] or test_ys[i][j] == 1:
                    bing += 1
                if test_ys[i][j] == 1:
                    yuanlai += 1
                if pred[i][j] > yuzhi[index]:
                    yuce += 1

            accuracy += jiao / bing
            if yuce==0:
                precision+=0
            else:
                precision += jiao / yuce
            recall += jiao / yuanlai

        accuracy = accuracy / 100
        precision = precision / 100
        recall = recall / 100
        if recall==0 and precision==0:
            f1_score=0
        else:
            f1_score = 2 * precision * recall / (precision + recall)

        print('accuracy:' + str(accuracy))
        print('precision:' + str(precision))
        print('recall:' + str(recall))
        print('f1_score:' + str(f1_score))

        fw.write(str(yuzhi[index]) + '\t' + str(accuracy) + '\t' + str(precision) + '\t' + str(recall) + '\t' + str(f1_score) + '\n')


def read_vocab():
    vocab = {}
    #build_vocab(train_dir, vocab_dir, 100000)
    with open(vocab_dir, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if (word not in vocab):
                vocab[word] = len(vocab) + 1
            if (len(vocab) + 1 >= 5000):
                break
    return vocab

yuzhi = [0.1,0.2,0.3,0.4,0.5]


#fw.write('\n\n\n' + str(len(categories)) + '\n')
now = datetime.datetime.strptime(time.strftime("%H:%M:%S"), "%H:%M:%S")
modelname=str(now)[11:19].replace(':','')
fw.write(modelname+'\n')
train_dir = 'data/0923/train.txt'
test_dir = 'data/0923/test1.txt'
val_dir = 'data/0923/val.txt'
vocab_dir = 'data/0923/vocab0.txt'
#build_vocab(train_dir, vocab_dir, 100000)
vocab=read_vocab()
train_xs, train_ys = load_data(train_dir)
train_data = [[] for i in range(2)]
for i in range(len(train_xs)):
    train_data[0].append(train_ys[i])
    train_data[1].append(train_xs[i])
val_xs, val_ys = load_data(val_dir)
test_xs, test_ys = load_data(test_dir)
train()

now = datetime.datetime.strptime(time.strftime("%H:%M:%S"), "%H:%M:%S")
modelname=str(now)[11:19].replace(':','')
fw.write(modelname+'\n')
train_dir = 'data/0923/train1.txt'
test_dir = 'data/0923/test1.txt'
val_dir = 'data/0923/val1.txt'
vocab_dir = 'data/0923/vocab1.txt'
#
vocab=read_vocab()
train_xs, train_ys = load_data(train_dir)
val_xs, val_ys = load_data(val_dir)
test_xs, test_ys = load_data(test_dir)
train_data = [[] for i in range(2)]
for i in range(len(train_xs)):
    train_data[0].append(train_ys[i])
    train_data[1].append(train_xs[i])
train()

now = datetime.datetime.strptime(time.strftime("%H:%M:%S"), "%H:%M:%S")
modelname=str(now)[11:19].replace(':','')
fw.write(modelname+'\n')
train_dir = 'data/0923/train.txt'
test_dir = 'data/0923/test2.txt'
val_dir = 'data/0923/val.txt'
vocab_dir = 'data/0923/vocab2.txt'
#build_vocab(train_dir, vocab_dir, 100000)
vocab=read_vocab()
train_xs, train_ys = load_data(train_dir)
val_xs, val_ys = load_data(val_dir)
test_xs, test_ys = load_data(test_dir)
train_data = [[] for i in range(2)]
for i in range(len(train_xs)):
    train_data[0].append(train_ys[i])
    train_data[1].append(train_xs[i])
train()

now = datetime.datetime.strptime(time.strftime("%H:%M:%S"), "%H:%M:%S")
modelname=str(now)[11:19].replace(':','')
fw.write(modelname+'\n')
train_dir = 'data/0923/train2.txt'
test_dir = 'data/0923/test2.txt'
val_dir = 'data/0923/val2.txt'
vocab_dir = 'data/0923/vocab3.txt'
#build_vocab(train_dir, vocab_dir, 100000)
vocab=read_vocab()
train_xs, train_ys = load_data(train_dir)
val_xs, val_ys = load_data(val_dir)
test_xs, test_ys = load_data(test_dir)
train_data = [[] for i in range(2)]
for i in range(len(train_xs)):
    train_data[0].append(train_ys[i])
    train_data[1].append(train_xs[i])
train()

fw.write('\n\n')
