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
categories = ["中医内科", "中医儿科", "中医外科", "中医妇科", "针灸推拿", "中医养生"]


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
    model = load_model(modelname)
    test_xs, test_ys = load_data("data/test.txt")
    test_ys = [t.argmax() for t in test_ys]
    pred = model.predict(test_xs).argmax(-1)



    acc_dict = {}

    print("Classification Report:")
    print(classification_report(test_ys, pred, target_names=categories, digits=4))
    print("Confusion Matrix:")
    cm = confusion_matrix(test_ys, pred)
    for i in range(len(cm[0])):
        tp = cm[i][i]
        tn = 0
        fp_fn = 0
        for j in range(len(cm)):
            if j != i:
                for k in range(len(cm)):
                    if k != i:
                        tn += cm[j][k]
                    else:
                        fp_fn += cm[j][k]
            else:
                for k in range(len(cm)):
                    if k != i:
                        fp_fn += cm[j][k]

        acc_dict[categories[i]] = ((tp + tn) / (tp + tn + fp_fn))
    print(acc_dict)
    print(cm)




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
test()

