import wfdb
import pywt
import matplotlib.pyplot as plt
import torch
import seaborn
import datetime
import numpy as np
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 项目目录
project_path = "D:\\A-TEST\\"
# 定义日志目录,必须是启动web应用时指定目录的子目录,建议使用日期时间作为子目录名
log_dir = project_path + "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + "ecg_model.h5"

PATH = 'af-termination-challenge-database-1.0.0/learning-set/'
ClassSet = ['n','s','t']
NumberSet = ['01','02','03','04','05','06','07','08','09','10']

#小波去噪预处理
def denoise(data):
    #用db5作为小波基，对心电数据进行9尺度小波变换，对系数处理
    coeffs = pywt.wavedec(data=data,wavelet='db5',level=9)
    cA9,cD9,cD8,cD7,cD6,cD5,cD4,cD3,cD2,cD1 = coeffs
    #使用软阈值滤波
    threshold =(np.median(np.abs(cD1))/0.6745)*(np.sqrt(2*np.log(len(cD1))))
    #将高频信号cD1、cD2置零
    cD1.fill(0)
    cD2.fill(0)
    #将其他中低频信号按软阈值公式滤波
    for i in range(1,len(coeffs)-2):
        coeffs[i]=pywt.threshold(coeffs[i],threshold)
    #小波反变换，获取去噪后的信号
    rdata=pywt.waverec(coeffs=coeffs,wavelet='db5')
    return rdata

x_data=[]
y_data=[]
train_data=[]

#切割
def cut_series(data,time,fs):
    cut_length=time*fs
    number=int(len(data)/cut_length)
    #print(number)
    for i in range(0,number):
        cut=data[i*cut_length:(i+1)*cut_length].reshape(-1,1)
        cut=MinMaxScaler(feature_range=(0,1)).fit_transform(cut)
        #print(cut)
        #plt.plot(cut)
        #plt.show()
        x_data.append(cut.flatten())

def read_single_data_cut(type,number):
    file_path=PATH+type+number
    record=wfdb.rdrecord(file_path,channels=[0])#读取
    rdata=record.p_signal.flatten()
    data=denoise(rdata)#去噪
    new_data=np.array(data)
    cut_series(new_data,10,128)#切割

def read_single_data(type,number):
    file_path=PATH+type+number
    record=wfdb.rdrecord(file_path,channels=[0])#读取
    rdata=record.p_signal.flatten()
    data=denoise(rdata)#去噪
    new_data=np.array(data)
    new_data=new_data.reshape(-1,1)
    new_data=MinMaxScaler(feature_range=(0,1)).fit_transform(new_data)
    #print(new_data)
    #plt.plot(new_data)
    #plt.show()
    x_data.append(new_data.flatten())
    

def read_all_data():
    for type in ClassSet:
        for number in NumberSet:
            read_single_data_cut(type,number)

#30个标签，前10个n组，后20个t组
for i in range(60):
    y_data.append("n")
for i in range(120):
    y_data.append("t")
y_data=np.array(y_data).reshape(-1,1)
#print(y_data)

read_all_data()
x_data=np.array(x_data)
#print(x_data.shape)
#print(x_data)

#训练集合并
train_data=np.hstack((x_data,y_data))
print(train_data.shape)
print(train_data)

#随机提取X为训练集，Y为标签集
np.random.shuffle(train_data)
X=train_data[:,:1280].reshape(-1, 1280, 1)
Y=train_data[:,1280]

RATIO=0.3
class_num=2

#混合打乱
shuffle_index=np.random.permutation(len(X))
test_length=int(RATIO*len(shuffle_index))
test_index=shuffle_index[:test_length]
train_index=shuffle_index[test_length:]
X_test,Y_test=X[test_index],Y[test_index]
X_train,Y_train=X[train_index],Y[train_index]

def buildModel():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1280,1)),
        # 第一个卷积层, 4 个 21x1 卷积核
        tf.keras.layers.Conv1D(filters=4,kernel_size=21,strides=1,padding='SAME',activation='relu'),
        # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3,strides=2,padding='SAME'),
        # 第二个卷积层, 16 个 23x1 卷积核
        tf.keras.layers.Conv1D(filters=16,kernel_size=23,strides=1,padding='SAME',activation='relu'),
        # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3,strides=2,padding='SAME'),
        # 第三个卷积层, 32 个 25x1 卷积核
        tf.keras.layers.Conv1D(filters=32,kernel_size=25,strides=1,padding='SAME',activation='relu'),
        # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.AvgPool1D(pool_size=3,strides=2,padding='SAME'),
        # 第四个卷积层, 64 个 27x1 卷积核
        tf.keras.layers.Conv1D(filters=64,kernel_size=27,strides=1,padding='SAME',activation='relu'),
        # 打平层,方便全连接层处理
        tf.keras.layers.Flatten(),
        # 全连接层,128 个节点
        tf.keras.layers.Dense(128,activation='relu'),
        # Dropout层,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # 全连接层,5 个节点
        tf.keras.layers.Dense(class_num,activation='softmax')
    ])
    return newModel

#混淆矩阵
def plotHeatMap(Y_test,Y_pred):
    con_mat=confusion_matrix(Y_test,Y_pred)
    #归一化
    #con_mat_norm=con_mat.astype('float')/con_mat.sum(axis=1)[:, np.newaxis]
    #con_mat_norm=np.around(con_mat_norm,decimals=2)
    #绘图
    plt.figure(figsize=(8,8))
    seaborn.heatmap(con_mat,annot=True,fmt='.20g',cmap='Blues')
    plt.ylim(0,5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

#构建CNN模型
model=buildModel()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
#定义tensorboard对象
#暂待优化
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#训练与验证
model.fit(X_train, Y_train, epochs=30,batch_size=50,validation_split=RATIO,callbacks=[tensorboard_callback])
# 预测
Y_pred = model.predict_classes(X_test)
# 绘制混淆矩阵
plotHeatMap(Y_test, Y_pred)



#np.savetxt("data.txt",x_data)
#注释是RR间期，单位1/128s
#annotation=wfdb.rdann('af-termination-challenge-database-1.0.0/learning-set/n01','qrs')
#print(annotation.sample)




  







     











