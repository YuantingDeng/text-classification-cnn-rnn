import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from models import *
import random
import os
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

vocab = {}
with open("data/vocab.txt", "r", encoding="utf-8") as f:
    for line in f:
        word = line.strip()
        if (word not in vocab):
            vocab[word] = len(vocab) + 1
        if (len(vocab) + 1 >= 5000):
            break

length = 500
categories = ["中医内科", "中医儿科", "中医外科", "中医妇科", "针灸推拿", "中医养生"]
batch=0

def batch_iter(data, batch_size=120):
    """生成批次数据"""

    l = [[] for j in range(6)]

    for i in range(len(data[0])):
        t = []
        t.append(data[0][i])
        t.append(data[1][i])
        l[data[0][i]].append(t)
    batch = int(len(l[0]) * 6 / batch_size)
    while True:
        temp = []
        x, y = [], []
        # print(len(l))
        # print(l[0])
        for j in range(len(l)):
            temp.extend(random.sample(l[j], 20))
        for each in temp:
            t=np.zeros(6,int)
            t[each[0]]=1
            x.append(each[1])
            y.append(t)
        x_shuffle = np.array(x)
        y_shuffle = np.array(y)
        indices = np.random.permutation(np.arange(len(x)))
        x_shuffle = x_shuffle[indices]
        y_shuffle = y_shuffle[indices]

        yield x_shuffle, y_shuffle


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

    return np.stack(xs), np.stack(ys)


train_xs, train_ys = load_data("data/train.txt")
train_data = [[] for i in range(2)]
for i in range(len(train_xs)):
    train_data[0].append(np.argmax(train_ys[i]))
    train_data[1].append(train_xs[i])
val_xs, val_ys = load_data("data/val.txt")

model = cnn_model(num_classes=len(categories), input_length=length)
# model = multi_cnn_model(num_classes=len(categories), input_length=length)
# model = rnn_model(num_classes=len(categories), input_length=length)
# model = crnnyuan_model(num_classes=len(categories), input_length=length)
# model = rcnn_model(num_classes=len(categories), input_length=length)
# model = crnn_model(num_classes=len(categories), input_length=length)
model.summary()
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# model.fit(
#     train_xs,
#     train_ys,
#     validation_data=(val_xs, val_ys),
#     shuffle=True,
#     epochs=2,
#     batch_size=64,
#     verbose=1
# )

model.fit_generator(batch_iter(train_data), steps_per_epoch=4400, validation_data=(val_xs, val_ys), epochs=2)
model.save("model.h5")

test_xs, test_ys = load_data("data/test.txt")
test_ys = [t.argmax() for t in test_ys]
pred = model.predict(test_xs).argmax(-1)
print("Classification Report:")
print(classification_report(test_ys, pred, target_names=categories, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(test_ys, pred))
