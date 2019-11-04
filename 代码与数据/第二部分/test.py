import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from models import *
import os
import datetime
import time
from keras.models import load_model
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

vocab_dir = 'data/vocab.txt'
modelname = 'modelcrnn.h5'
#fw = open('test.txt', 'a+', encoding='utf-8')


length = 500
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

            c = categories.index(category)
            ys.append(keras.utils.to_categorical(c, len(categories)))

            # words = category.strip().split(' ')
            # c = np.zeros(len(categories))
            # for word in words:
            #     c[categories.index(word)] = 1
            # ys.append(c)

    return np.stack(xs), np.stack(ys)


def test():
    #model_dir = 'model/'+ modelname+'.h5'
    model = load_model('modelcrnn.h5')
    pred = model.predict(test_xs)

    # 指标二
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

        #fw.write(str(yuzhi[index]) + '\t' + str(accuracy) + '\t' + str(precision) + '\t' + str(recall) + '\t' + str(f1_score) + '\n')





def read_vocab():
    vocab = {}
    # build_vocab(train_dir, vocab_dir, 100000)
    with open(vocab_dir, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if (word not in vocab):
                vocab[word] = len(vocab) + 1
            if (len(vocab) + 1 >= 5000):
                break
    return vocab

vocab = read_vocab()

yuzhi = [0.1,0.2,0.3,0.4,0.5]


train_dir = 'data/train.txt'
test_dir = 'data/test1.txt'
val_dir = 'data/val.txt'
vocab_dir = 'data/vocab0.txt'
modelname='12_20_57'
vocab=read_vocab()
train_xs, train_ys = load_data(train_dir)
val_xs, val_ys = load_data(val_dir)
test_xs, test_ys = load_data(test_dir)
test()
#
# train_dir = 'data/0923/train1.txt'
# test_dir = 'data/0923/test1.txt'
# val_dir = 'data/0923/val1.txt'
# vocab_dir = 'data/0923/vocab1.txt'
# modelname='12_00_42'
# vocab=read_vocab()
# train_xs, train_ys = load_data(train_dir)
# val_xs, val_ys = load_data(val_dir)
# test_xs, test_ys = load_data(test_dir)
# test()
#
# train_dir = 'data/0923/train.txt'
# test_dir = 'data/0923/test2.txt'
# val_dir = 'data/0923/val.txt'
# vocab_dir = 'data/0923/vocab2.txt'
# modelname='12_02_18'
# vocab=read_vocab()
# train_xs, train_ys = load_data(train_dir)
# val_xs, val_ys = load_data(val_dir)
# test_xs, test_ys = load_data(test_dir)
# test()
#
# train_dir = 'data/0923/train2.txt'
# test_dir = 'data/0923/test2.txt'
# val_dir = 'data/0923/val2.txt'
# vocab_dir = 'data/0923/vocab3.txt'
# modelname='12_21_45'
# vocab=read_vocab()
# train_xs, train_ys = load_data(train_dir)
# val_xs, val_ys = load_data(val_dir)
# test_xs, test_ys = load_data(test_dir)
# test()

