# -*- coding: utf-8 -*-
# Author: SHI ZHONGMIAO  
# TIME:2022/8/10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf

x_data=datasets.load_iris().data
y_data=datasets.load_iris().target
# 乱序
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)

# 划分测试和训练
x_train=x_data[:-30]
y_train=y_data[:-30]
x_test=x_data[-30:]
y_test=y_data[-30:]

#数据类型转换
x_train=tf.cast(x_train,tf.float32)
x_test=tf.cast(x_test,tf.float32)
#数据匹配
train_db=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)


#定义神经网络训练参数
w1=tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1)) #4输入3输出
b1=tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))
#定义超参数
lr=0.1 #学习率
train_loss_results=[] #将每轮loss记录在次列表中，为后续话loss提供依据
test_acc=[]
epoch=500 #循环次数500
loss_all=0

#循环嵌套迭代
for epoch in range(epoch):#数据集别迭代
    for step,(x_train,y_train) in enumerate(train_db):
        with tf.GradientTape() as tape: #记录梯度信息
            y=tf.matmul(x_train, w1) + b1
            y=tf.nn.softmax(y) #使y符合概率分布
            y_=tf.one_hot(y_train,depth=3) #将标签数字转化为独热编码格式
            loss=tf.reduce_mean(tf.square(y_-y))#均方差损失函数
            loss_all +=loss.numpy()#对每次的损失值进行lei加
        #计算loss中各参数的梯度
        grads=tape.gradient(loss,[w1,b1])
        w1.assign_sub(lr*grads[0]) #参数自更新
        b1.assign_sub(lr*grads[1]) # canshu1zigengxin1

    #打印每个epoch
    print('Epoch{},loss{}'.format(epoch,loss_all/4))
    train_loss_results.append(loss_all/4)
    loss_all=0 #为下一个做准备

##
#TEST PART
    total_correct,total_number=0,0
    for x_test,y_test in test_db:
        y=tf.matmul(x_test,w1)+b1
        y=tf.nn.softmax(y)
        pred=tf.argmax(y,axis=1)
        #将pred 转化为Ytest数据类型
        pred=tf.cast(pred,dtype=y_test.dtype)
        #弱分类正确则correct=1否者为0,将bool转化为int型
        correct=tf.cast(tf.equal(pred,y_test),dtype=tf.int32)
        correct=tf.reduce_sum(correct)
        total_correct +=int(correct)
        total_number +=x_test.shape[0]
    acc=total_correct/total_number
    test_acc.append(acc)
    print('Test _acc',acc)
    print('*/'*50)
plt.title('Loss Function Curive')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.plot(train_loss_results,label='$loss$')
plt.legend()
plt.show()
#话精确度曲线
plt.title('ACC Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc,label='$Accuracy$')
plt.legend()
plt.show()