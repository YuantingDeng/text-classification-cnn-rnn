# f=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/else.txt','w',encoding='utf-8')
# f1=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/else1.txt','r',encoding='utf-8')
# f2=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/else2.txt','r',encoding='utf-8')
# f3=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/else3.txt','r',encoding='utf-8')
# f4=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/else4.txt','r',encoding='utf-8')
#
# a,a1,a2,a3,a4=[],[],[],[],[]
# for line in f1:
#     a1.append(line)
# for line in f2:
#     a2.append(line)
# for line in f3:
#     a3.append(line)
# for line in f4:
#     a4.append(line)
#
# for i in a1:
#     if i in a2 and i in a3 and i in a4:
#         f.write(i)


# for i in a1:
#     if i in a2 and i in a3 and i in a4:
#         a.append(i)
# f=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/data.txt','r',encoding='utf-8')
# for i in a:
#     f.write(i)

# f1=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/80000cnncorrect.txt','r',encoding='utf-8')
# f2=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/80000rnncorrect.txt','r',encoding='utf-8')
# #f3=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/80000crnncorrect.txt','r',encoding='utf-8')
# f4=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/80000rcnncorrect.txt','r',encoding='utf-8')
# a,a1,a2,a3,a4=[],[],[],[],[]
# for line in f1:
#     a1.append(line)
# for line in f2:
#     a2.append(line)
# i=1
# for line in f4:
#     if i<51091:
#         a3.append(line)
#     else:
#         a4.append(line)
#     i=i+1
#
# for i in a1:
#     if i in a2 and i in a3 and i in a4:
#         a.append(i)
# f=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/80000correct.txt','w',encoding='utf-8')
# for i in a:
#     f.write(i)

# f1=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/80000cnnwrong.txt','r',encoding='utf-8')
# f2=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/80000rnnwrong.txt','r',encoding='utf-8')
# #f3=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/80000crnnwrong.txt','r',encoding='utf-8')
# f4=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/80000rcnnwrong.txt','r',encoding='utf-8')
# a,a1,a2,a3,a4=[],[],[],[],[]
# for line in f1:
#     a1.append(line)
# for line in f2:
#     a2.append(line)
# i=1
# for line in f4:
#     if i <54460:
#         a3.append(line)
#     else:
#         a4.append(line)
#     i=i+1
# for i in a1:
#     if i in a2 and i in a3 and i in a4:
#         a.append(i)
# f=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/80000wrong.txt','w',encoding='utf-8')
# for i in a:
#     f.write(i)

# f1=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/test.txt','r',encoding='utf-8')
# f2=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/x.txt','r',encoding='utf-8')#crnn
# f3=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/y.txt','r',encoding='utf-8')
# f4=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/data.txt','w',encoding='utf-8')
# x,y,test=[],[],[]
# xcorret=[]
# for line in f1:
#     test.append(line)
# for line in f2:
#     data=line.strip().split('\t',3)
#     xcorret.append(data[0])
#     x.append(data[1]+'\t'+data[2]+'\n')#crnn
# for line in f3:
#     y.append(line)
#
# xrest,yrest=[],[]
# z=[]
# f4=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/data.txt','w',encoding='utf-8')
# f5=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/xrest.txt','r',encoding='utf-8')
# f6=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/yrest.txt','r',encoding='utf-8')
# for line in f5:
#     xrest.append(line)
# for line in f6:
#     yrest.append(line)
# for i in x:
#     if i in yrest:
#         z.append(i)
# zz=[]
# for i in range(int(len(z)*0.8)):
#     zz.append(z[i])
# flag=0
# for i in range(len(test)):
#     flag=0
#     for j in range(len(zz)):
#         if test[i]==zz[j]:
#             lz = test[i].strip().split('\t')
#             f4.write(xcorret[j] + '\t' + lz[1] + '\n')
#             flag=1
#             break
#     if flag==0:
#         f4.write(test[i])


# f1=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/data36_6class.txt','r',encoding='utf-8')
# f2=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/test.txt','r',encoding='utf-8')
# f=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/data17.txt','w',encoding='utf-8')
# data36_6class,test=[],[]
# for i in f1:
#     j=i.strip().split('\t',3)
#     data36_6class.append(j)
# for i in f2:
#     j = i.strip().split('\t', 2)
#     test.append(j)
# for i in range(len(test)):
#     for j in range(len(data36_6class)):
#         if len(test[i])==2 and len(data36_6class[j])==3:
#             if test[i][1]==data36_6class[j][1]:
#                 f.write(test[i][0]+'\t'+test[i][1]+'\t'+data36_6class[j][2]+'\n')


f1=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/test.txt','r',encoding='utf-8')
f2=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/x.txt','r',encoding='utf-8')#crnn
f3=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/y.txt','r',encoding='utf-8')
f5=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/xrest.txt','r',encoding='utf-8')
f6=open('F:/毕业/text-classification-cnn-rnn-master/data/cnews/604cnn/yrest.txt','r',encoding='utf-8')
x=[]
for i in f5:
    x.append(i)
y=0
for i in f3:
    data = i.strip().split('\t', 3)
    line=data[1]+'\t'+data[2]+'\n'
    #print(line)
    if line in x:
        y=y+1
        print(line)
print(y)