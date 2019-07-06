# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
import jieba
import random
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)
    print(len(data_train))
    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)

    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    #categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = ['中医内科', '中医外科', '中医妇科', '中医儿科', '中医养生', '针灸推拿']
    #categories = ['中医内科',  '中医妇科', '中医儿科', '针灸推拿']
    #categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def create_file(filename,train_dir,test_dir,val_dir,train_per_class_size,val_per_class_size):
    ftrain=open(train_dir,'w',encoding='utf-8')
    #ftest = open(test_dir, 'w', encoding='utf-8')
    fval = open(val_dir, 'w', encoding='utf-8')
    data=[]
    train=[]
    val=[]
    test=[]
    train_class_num=[0,0,0,0,0,0]
    val_class_num=[0,0,0,0,0,0]
    category,_=read_category()
    with open_file(filename) as f:
        for line in f:
            data.append(line)
    random.shuffle(data)
    for i in range(len(data)):
        question=data[i].strip().split('\t')
        for j in range(len(category)):
            if question[0]==category[j] and train_class_num[j]<train_per_class_size:
                ftrain.write(data[i])
                train_class_num[j]=train_class_num[j]+1
                break
            elif question[0]==category[j] and val_class_num[j]<val_per_class_size:
                fval.write(data[i])
                val_class_num[j]=val_class_num[j]+1
                break
        #ftest.write(data[i])

def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def tfidf(filename,k_rate):

    question_list,lables = [],[]
    c=1
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            a=line.strip().split('\t')
            if(len(a)==2):
                lables.append(a[0])
                question_list.append(a[1])


    for i in range(len(question_list)):
        question=jieba.lcut(question_list[i])
        q=""
        for j in range(len(question)):
            q=q+question[j]+" "
        question_list[i]=q.strip()
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(question_list))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    #weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    fw=open("./cnews/531/test.txt",'a+',encoding="utf-8")
    for i in range(len(question_list)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        if i > 120473:
            t = tfidf[i].toarray()
            #print(t)
            fw.write(lables[i]+'\t')
            for j in range(len(word)):
                if t[0][j]>k_rate:
                    #print(word[j])
                    fw.write(word[j])
                    #print(word[j], weight[i][j])
            fw.write('\n')

filename = './cnews/531/cnews.test.txt'
#
#tfidf(filename,0.05)