#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from crnn_model import TCRNNConfig, TextCRNN
#from cnn import CNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab,read_file,create_file
#from data.cnews_loader2 import read_vocab, read_category, batch_iter, process_file, build_vocab

base_dir = 'data/cnews/604cnn'
data_dir=os.path.join(base_dir, 'data.txt')
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'test.txt')
val_dir = os.path.join(base_dir, 'val.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir1 = 'checkpoints/textcnn'
save_path1 = os.path.join(save_dir1, 'best_validation')  # 最佳验证结果保存路径
save_dir2 = 'checkpoints/textcrnn'
save_path2 = os.path.join(save_dir2, 'best_validation')  # 最佳验证结果保存路径
save_path=[]
save_path.append(save_path1)
save_path.append(save_path2)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train(x):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    print('train num'+str(len(x_train)))

    config1 = tf.ConfigProto(allow_soft_placement=True)

    # 最多占gpu资源的70%
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    # 开始不会给tensorflow全部gpu资源 而是按需增加
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 500  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path[x])
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test(x):
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)
    config1 = tf.ConfigProto(allow_soft_placement=True)

    # 最多占gpu资源的70%
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    # 开始不会给tensorflow全部gpu资源 而是按需增加
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path[x])  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
    f=open('result/cnn/resultcnn.txt','a+',encoding='utf-8')
    f.write(str(acc_test)+'\n')
    f.write(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
    f.write('\n')
    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    if x==1:
        xx=[]
        f=open(test_dir,'r',encoding='utf-8')
        for line in f:
            zx=line.strip().split('\t')
            if len(zx)==2:
                xx.append(line)
        f.close()
        print(len(xx))
        print(len(y_test_cls))
        else_dir=base_dir+'/x.txt'
        else_dir2=base_dir+'/xrest.txt'
        fw=open(else_dir,'w',encoding='utf-8')
        fw2 = open(else_dir2, 'w', encoding='utf-8')
        for i in range(len(y_test)):
            if y_pred_cls[i]!=y_test_cls[i]:
                fw.write(categories[y_pred_cls[i]]+'\t'+xx[i])
            else:
                fw2.write(xx[i])
    else:
        xx=[]
        f=open(test_dir,'r',encoding='utf-8')
        for line in f:
            zx=line.strip().split('\t')
            if len(zx)==2:
                xx.append(line)
        f.close()
        print(len(xx))
        print(len(y_test_cls))
        else_dir=base_dir+'/y.txt'
        else_dir2=base_dir+'/yrest.txt'
        fw=open(else_dir,'w',encoding='utf-8')
        fw2 = open(else_dir2, 'w', encoding='utf-8')
        for i in range(len(y_test)):
            if y_pred_cls[i]!=y_test_cls[i]:
                fw.write(categories[y_pred_cls[i]]+'\t'+xx[i])
            else:
                fw2.write(xx[i])



if __name__ == '__main__':
    # if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
    #     raise ValueError("""usage: python run_cnn.py [train / test]""")
    #choice=input("train or test:")
    # if choice=='train':
    #     create_file(data_dir,train_dir,test_dir,val_dir,4000,1000)

    print('Configuring CNN model...')

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, 100000)

    filter_sizes = [3, 4,5]  # 3
    num_filters = 32
    #model = CNN(config.seq_length,config.num_classes,config.vocab_size,config.embedding_dim,filter_sizes,num_filters,0.0)


    for i in range(1,11):
        if i%2!=0:
            config = TCNNConfig()
            categories, cat_to_id = read_category()
            words, word_to_id = read_vocab(vocab_dir)
            config.vocab_size = len(words)
            create_file(data_dir, train_dir, test_dir, val_dir, 4000, 1000)
            build_vocab(train_dir, vocab_dir, 100000)
            model = TextCNN(config)
            train(0)
            test(0)
        else:
            config = TCRNNConfig()
            categories, cat_to_id = read_category()
            words, word_to_id = read_vocab(vocab_dir)
            config.vocab_size = len(words)
            create_file(data_dir, train_dir, test_dir, val_dir, 4000, 1000)
            build_vocab(train_dir, vocab_dir, 100000)
            model = TextCRNN(config)
            train(1)
            test(1)
        if i%2==0:
            f1=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/test.txt','r',encoding='utf-8')
            f2=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/x.txt','r',encoding='utf-8')#crnn
            f3=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/y.txt','r',encoding='utf-8')
            f4=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/data.txt','w',encoding='utf-8')
            x,y,test=[],[],[]
            xcorret=[]
            for line in f1:
                test.append(line)
            for line in f2:
                data=line.strip().split('\t',3)
                xcorret.append(data[0])
                x.append(data[1]+'\t'+data[2]+'\n')#crnn
            for line in f3:
                y.append(line)
            xrest,yrest=[],[]
            z=[]

            f5=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/xrest.txt','r',encoding='utf-8')
            f6=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/yrest.txt','r',encoding='utf-8')
            for line in f5:
                xrest.append(line)
            for line in f6:
                yrest.append(line)
            for i in x:
                if i in yrest:
                    z.append(i)
            zz=[]
            for i in range(int(len(z)*0.8)):
                zz.append(z[i])
            flag=0
            for i in range(len(test)):
                flag=0
                for j in range(len(zz)):
                    if test[i]==zz[j]:
                        lz = test[i].strip().split('\t')
                        f4.write(xcorret[j] + '\t' + lz[1] + '\n')
                        flag=1
                        break
                if flag==0:
                    f4.write(test[i])