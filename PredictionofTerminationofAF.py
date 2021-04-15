import wfdb
import pywt
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler

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
        print(cut)
        plt.plot(cut)
        plt.show()
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
            read_single_data(type,number)

#30个标签，前10个n组，后20个t组
for i in range(10):
    y_data.append("n")
for i in range(20):
    y_data.append("t")
y_data=np.array(y_data).reshape(-1,1)
#print(y_data)

read_all_data()
x_data=np.array(x_data)
print(x_data.shape)
print(x_data)

#训练集合并
train_data=np.hstack((x_data,y_data))
print(train_data.shape)
print(train_data)

#np.savetxt("data.txt",x_data)


#record=wfdb.rdrecord('af-termination-challenge-database-1.0.0/learning-set/n01',channels=[0])

#annotation=wfdb.rdann('af-termination-challenge-database-1.0.0/learning-set/n01','qrs')
#print(annotation.sample)
#print(annotation.symbol)



  







     











