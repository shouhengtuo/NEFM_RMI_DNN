# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 08:57:25 2021

@author: Shouheng Tuo
"""
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
from tensorboardX import SummaryWriter
import numpy as np
from thop import profile
from graphviz import Digraph
from torch.utils import data # 获取迭代数据
from matplotlib.font_manager import FontProperties
from scipy import stats
import seaborn as sns
import os
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10)



## 载入数据



Region = np.array(['安徽','北京','福建','甘肃','广东','广西','贵州','海南','河北','河南','黑龙江','湖北','湖南','吉林','江苏','江西','辽宁','内蒙古','宁夏','青海','山东','山西','陕西','上海',
                   '四川','天津','西藏','新疆','云南','浙江','重庆'])
Eindex = np.array(['1第一产业增加值占GDP比重','2第二产业增加值占GDP比重','3第三产业增加值占GDP比重','GDP','GDP增长指数(上年＝100)','城镇居民消费水平','城镇居民消费水平指数(上年=100)','第二产业增加值',
                   '第二产业增加值增长指数(上年＝100)','第三产业增加值','第三产业增加值增长指数(上年＝100)'	,'第一产业增加值','第一产业增加值增长指数(上年＝100)','工业增加值','行业增加值_房地产业',
                   '行业增加值_建筑业','行业增加值_交通运输、仓储和邮政业','行业增加值_金融业','行业增加值_农、林、牧、渔业'	,'行业增加值_批发和零售业','行业增加值_住宿和餐饮业','居民消费水平',
                   '居民消费水平指数(上年=100)','农村居民消费水平','农村居民消费水平指数(上年=100)','人均GDP','人均GDP增长指数(上年＝100)'])
fdata0 = pd.read_csv("data.csv",encoding='utf-8')
fdata = np.array(fdata0)
fdata2 = np.zeros(shape=(27,18)) ## 27个指标，2000-2017 （18年数据），每次提取一个省的数据，
fdataT = np.zeros(shape=(31,18,27)) ## 转置矩阵
Xdata = np.zeros(shape=(31*18,27)) ## 将所有省份相同指标数据 拼接在同一列
for i in range(31):
    begin = i * 27
    end = begin + 27
    k = 0
    for j in range(begin, end):        
        a = np.array(fdata[j])[0].split("\t")
        fdata2[k,:] = np.array(a[4:],dtype="float")
        k = k + 1
    dataT = fdata2.T
    fdataT[i,:,:] = dataT
    
Xdata = fdataT.reshape(Xdata.shape)  
maxValue =  1.1 * np.max(Xdata,0)

## 归一化数据
Xdata2 = np.zeros(Xdata.shape)
folder = ".\\数据分布图\\正态化之前"
for i in range(27):
    Xdata2[:,i] = Xdata[:,i]/maxValue[i]
    
  
    # ## 绘制  数据分布图
    # plt.figure()
    # sns.distplot(Xdata2[:,i])
    # plt.title('{}--分布图'.format(Eindex[i]),fontproperties=font)
    # plt.ylabel('数量',fontproperties=font)
    # plt.xlabel('数值',fontproperties=font)
    # filename = folder + '\\{}_hist.tiff'.format(Eindex[i])
    # plt.savefig(filename, dpi=600,format='tiff')
    # filename = folder + '\\{}_hist.jpg'.format(Eindex[i])
    # plt.close()
    # fig, ax = plt.subplots(figsize=[12,8])
    # prob = stats.probplot(Xdata2[:,i], dist=stats.norm,plot=ax)
    # plt.ylabel('有序值 Order Values',fontproperties=font)
    # plt.xlabel('理论分位数 Theoretical quantiles',fontproperties=font)
    # plt.title('{}--QQ图'.format(Eindex[i]),fontproperties=font)
    # filename = folder + '\\{}_QQ.tiff'.format(Eindex[i])
    # plt.savefig(filename, dpi=600,format='tiff')
    # plt.close()

    
"""   
   如果数据不满足正态分布，进行Box Cox 转换
   y = (x^lambda-1)/lambda  (lambda in (-0.5,0.5))

检验数据是否服从正态分布

kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
结果返回两个值：statistic → D值，pvalue → P值
p值大于0.05，为正态分布
H0:样本符合  
H1:样本不符合 
如果p>0.05接受H0 ,反之 
"""
Lambda = np.linspace(-0.99, 0.99,100) 
myLambda = np.zeros(27)
myPvalue = np.zeros(27)
Xdata3 = np.zeros(Xdata2.shape)
for i in range(27):
    flag = 0
    p0 = 0
    for lamd in Lambda:
        x = (1-np.power(Xdata2[:,i],lamd))/lamd
        s = pd.DataFrame(x,columns = ['value'])
        u = s['value'].mean()  # 计算均值
        std = s['value'].std()  # 计算标准差
        st,pvalue = stats.kstest(s['value'],'norm',(u,std))   
        if pvalue>0.001 and pvalue > p0:
            myLambda[i] = lamd
            myPvalue[i] = pvalue
            p0 = pvalue
            Xdata3[:,i] = x
            flag = 1
    if flag == 0:
        Xdata3[:,i] = Xdata2[:,i]
        print("not find a lambda for ",i)
        print(i,pvalue)

xind = np.array([6,12,13,18] )

for i in range(len(xind)):   
    x0 = Xdata3[:,xind[i]]
    
    x = 1/np.power(x0,0.2)
    Xdata3[:,xind[i]] = x ####***
    s = pd.DataFrame(x,columns = ['value'])
    u = s['value'].mean()  # 计算均值
    std = s['value'].std()  # 计算标准差
    st,pvalue = stats.kstest(s['value'],'norm',(u,std))
    print(i+1,pvalue)
    plt.figure()
    plt.hist(x)
    
    fig = plt.figure()
    res = stats.probplot(x0, plot=plt) #默认检测是正态分布
    plt.ylabel('有序值 Order Values')
    plt.xlabel('理论分位数 Theoretical quantiles')
    plt.title('初始概率图')
    plt.show()
    
    fig = plt.figure()
    res = stats.probplot(x0, plot=plt) #默认检测是正态分布
    plt.ylabel('有序值 Order Values')
    plt.xlabel('理论分位数 Theoretical quantiles')
    plt.title('预处理后概率图')
    plt.show()

"""
## 绘制 正态化之后 数据分布图
folder = ".\\数据分布图\\正态化之后"
for i in range(27):
    plt.figure()
    sns.distplot(Xdata3[:,i])
    plt.title('{}--分布图'.format(Eindex[i]),fontproperties=font)
    plt.ylabel('数量',fontproperties=font)
    plt.xlabel('数值',fontproperties=font)
    filename = folder + '\\{}_hist.tiff'.format(Eindex[i])
    plt.savefig(filename, dpi=600,format='tiff')
    filename = folder + '\\{}_hist.jpg'.format(Eindex[i])
    plt.close()
    fig, ax = plt.subplots(figsize=[12,8])
    prob = stats.probplot(Xdata3[:,i], dist=stats.norm,plot=ax)
    plt.ylabel('有序值 Order Values',fontproperties=font)
    plt.xlabel('理论分位数 Theoretical quantiles',fontproperties=font)
    plt.title('{}--QQ图'.format(Eindex[i]),fontproperties=font)
    filename = folder + '\\{}_QQ.tiff'.format(Eindex[i])
    plt.savefig(filename, dpi=600,format='tiff')
    plt.close() 
"""
"""
检验数据是否服从正态分布

kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
结果返回两个值：statistic → D值，pvalue → P值
p值大于0.05，为正态分布
H0:样本符合  
H1:样本不符合 
如果p>0.05接受H0 ,反之 

 
    s = pd.DataFrame(x,columns = ['value'])
    u = s['value'].mean()  # 计算均值
    std = s['value'].std()  # 计算标准差
    st,pvalue = stats.kstest(s['value'],'norm',(u,std))
"""

# 输入尺寸
row = 3
col = 27
outNum = 27
outIndex = [i for i in range(27)]  # 输出的指标编号
snum = 465
Xtrain = np.empty((snum,1,row,col))
Ytrain = np.empty((snum,outNum))
x0 = np.empty((5*snum,1,row,col)) # 随机样本
y0 = np.empty((5*snum,outNum))
T = 18 # 共18年数据
c = 0
c2 = 0

for i in range(31):
    for t in range(T-row):
        sb = i*T + t  
        se = sb + row  #连续3年数据
        # print(sb)
        # print(se)
        x = np.array(Xdata3[sb:se,:]);
        y = np.array(Xdata3[se,outIndex]);
        Xtrain[c] = x;
        Ytrain[c] = y; 
        
        ## 增加样本量 x0,y0 存放随机样本
        for k in range(5):
            x0[c2] = Xtrain[c] * ((np.random.rand(1,row,col)-0.5) * 0.0001 + 1) 
            y0[c2] = Ytrain[c] * (1 + (np.random.rand(1,outNum)-0.5) * 0.0001)
            c2 = c2 + 1
        c = c + 1
# Xtrain，Ytrain 中存放的是原始样本数据（归一化，正态化后的数据）
# Xtrain0,Ytrain0 中增加了 随机扰动的样本数据        
Xtrain0 = np.vstack([Xtrain,x0])
Ytrain0 = np.vstack([Ytrain,y0])
        
train_dataset0 = data.TensorDataset(torch.Tensor(Xtrain0),torch.Tensor(Ytrain0))    
train_loader0 = data.DataLoader(train_dataset0,batch_size=30,shuffle=True)

train_dataset = data.TensorDataset(torch.Tensor(Xtrain),torch.Tensor(Ytrain))   
       
train_loader = data.DataLoader(train_dataset,batch_size=40,shuffle=True)

test_loader = data.DataLoader(train_dataset,batch_size=15,shuffle=False)          
        

## 自定义损失函数

class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x,y):        
        return torch.sum(torch.log(torch.cosh((x-y))))
    
## 定义CNN network
class CNNnetwork(torch.nn.Module):
    def __init__(self):
        super(CNNnetwork,self).__init__()
        self.conv1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, 
                            out_channels=64, 
                            kernel_size=(2,1),
                            stride = 1,
                            padding = (1,1)),
            # torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,2))
            )
        self.conv1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, 
                            out_channels=128, 
                            kernel_size=(1,3),
                            stride = 1,
                            padding = (1,1)),
            # torch.nn.BatchNorm2d(32),
            # torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(1,2))            
            )
     
        
        self.conv1_3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, 
                            out_channels=256, 
                            kernel_size=(2,2),
                            stride = 1,
                            padding = (1,1)),
            # torch.nn.BatchNorm2d(16),
            # torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,2))
            )
        self.conv1_4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, 
                            out_channels=128, 
                            kernel_size=(2,2),
                            stride = 1,
                            padding = (0,0)),
            # torch.nn.BatchNorm2d(16),
            # torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(1,2))
            )
      
        
        inSize = 128
        hideSize = 128
        outSize = 27
                
        self.mlp1_1 = torch.nn.Sequential(
            torch.nn.Linear(inSize, hideSize )
            # torch.nn.Dropout(0.5),
            # torch.nn.Sigmoid()
            )
        
        self.mlp1_2 = torch.nn.Sequential(
            torch.nn.Linear(hideSize , outSize)
            # torch.nn.BatchNorm1d(outSize),
            # torch.nn.Dropout(0.5),
            # torch.nn.Sigmoid()
            )
        # self.dropout = torch.nn.Dropout(0.5)
        
        
        
    def forward(self,x):   
        xsize = x.shape
               
        x0 = self.conv1_1(x)
        
        x0 = self.conv1_2(x0)
        
        x0 = self.conv1_3(x0)
        
        x0 = self.conv1_4(x0)
        
        x0 = torch.squeeze(x0)         
        
        x1 = self.mlp1_1(x0)  
             
        x1 = self.mlp1_2(x1)
      
                    
        return x1
    

model = CNNnetwork()
use_GPU = torch.cuda.is_available()


loss_func = torch.nn.SmoothL1Loss()
# loss_func = torch.nn.MSELoss()
loss_func = My_loss()

# loss_func = torch.nn.MSELoss(reduction='sum')
if(use_GPU):
    model = model.cuda()
    # loss_func = My_loss()
    # loss_func = loss_func.cuda()
    # loss_func1 = loss_func1.cuda()

opt2 = torch.optim.AdamW(model.parameters(),lr=0.001,betas=(0.9,0.999) , eps=1e-8)
loss_count = []

for epoch in range(800):
   model.train() 
   for i,(x,y) in enumerate(train_loader0):
        batch_x = Variable(x)
        batch_y = Variable(y)
        if(use_GPU):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
               
        y1= model(batch_x)
        loss = loss_func(y1,batch_y)        
        opt2.zero_grad()       
        loss.backward()                  
        opt2.step()
        if i%20 == 0:
            loss_count.append(loss)
            
            # loss4_count.append(loss4)
            print('epoch{}:{}\t'.format(epoch,i), loss.item())
            torch.save(model,r'.\log_CNN')
            
plt.figure('PyTorch_CNN_Loss')
plt.semilogy(loss_count,label='Loss')
plt.legend()
plt.show()

plt.figure('predict values1')

plt.plot(y1.detach().cpu().numpy(),'bo-')
plt.plot(batch_y.detach().cpu().numpy(),'r*-', label="loss1")
plt.show()

plt.figure()
plt.plot(y1[10,:].detach().cpu().numpy(),'bo-')
plt.plot(batch_y[10,:].detach().cpu().numpy(),'r*-', label="loss1")
plt.show()

# 展示每个区域的预测情况
matplotlib.rcParams['font.sans-serif']=['SimHei']  #使用指定的汉字字体类型（此处为黑体）
xtime = torch.linspace(2003,2017,15)
MSEDATA = np.zeros((31,outNum))

R2 = np.zeros((31,outNum))

RMSLE = np.zeros((31,outNum))  #(Root Mean Squared Logarithmic Error)
Ynew = torch.zeros((5,1,1,27))

RegionFurture = torch.zeros((31,5,27))

for i,(x,y) in enumerate(test_loader):
    batch_x = Variable(x)
    batch_y = Variable(y)
    if(use_GPU):
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
    y = model(batch_x)
    
     # 还原数据 
    by1 = torch.zeros(batch_y.shape)
    y1 = torch.zeros(y.shape)
    for j in range(outNum): 
        if j in xind:
            # 反向还原数据：y = (1/x)^5
            by1[:,j] = torch.pow( (1 / batch_y[:,j]) , 5)
            y1[:,j] = torch.pow( (1 / y[:,j]), 5)
        else:
            #1. 反向 BOX-COX 操作 y = (1 - lambda*x)^(1/lambda)
            by1[:,j] = torch.pow( (1 - myLambda[j]*batch_y[:,j]) , 1/myLambda[j])
            y1[:,j] = torch.pow( (1-myLambda[j]*y[:,j]), 1/myLambda[j])
        #. 反向归一化操作
        by1[:,j] = by1[:,j]*maxValue[j]
        y1[:,j] = y1[:,j]*maxValue[j]
    
        a = torch.sum(torch.pow(y1[:,j] - by1[:,j],2))
        b = torch.sum(torch.pow(torch.mean(by1[:,j]) - by1[:,j],2))
        R2[i,j] = 1 - a/b
        RMSLE[i,j] =  torch.sqrt(torch.mean(torch.pow( torch.log(y1[:,j]+1) - torch.log(by1[:,j] + 1),2)))
        
       
        """
        if os.path.exists('.\\figures\\{}'.format(Region[i]))==False:
            os.makedirs('.\\figures\\{}'.format(Region[i]))
        for k in range(27):
            plt.figure()
            plt.title('{}.{}'.format(Region[i],Eindex[k]),FontProperties=font)
            plt.plot(xtime.detach().cpu().numpy(),y1[:,k].detach().cpu().numpy(),'bo-',label='预测值')
            plt.plot(xtime.detach().cpu().numpy(),by1[:,k],'r*-',label='真实值')
            plt.xlabel('年',FontProperties=font)
            plt.xticks(xtime,xtime.cpu().numpy().astype(int),rotation = 90)
            plt.ylabel('增长值',FontProperties=font)
            plt.legend()
            # plt.ylim((-0.1,1))
            filename = '.\\figures\\{}\\{}.jpg'.format(Region[i],Eindex[k])
            plt.savefig(filename)
            plt.close()
    # numpy.savetxt(".\\resultsData\\{}.txt".format(Region[i]),trainMat,fmt="%s",delimiter=",")
    """
        #预测未来5年发展
    xnew = batch_y[12:15,:]
    xnew = torch.reshape(xnew,(1,1,3,27))
    
    for k in range(5):
        Ynew[k] = model(xnew)
        y2 = torch.reshape(Ynew[k],(1,27))
        xnew = torch.cat((xnew[0,0,1:3,0:27],y2.cuda()),0)
        xnew = torch.reshape(xnew,(1,1,3,27))
        
    Ynew2 = torch.squeeze(Ynew)    
    for j in range(outNum): 
        if j in xind:
            # 反向还原数据：y = (1/x)^5
            Ynew2[:,j] = torch.pow( (1 / Ynew2[:,j]) , 5)
        else:
            #1. 反向 BOX-COX 操作 y = (1 - lambda*x)^(1/lambda)
            Ynew2[:,j] = torch.pow( (1 - myLambda[j]*Ynew2[:,j]) , 1/myLambda[j])
            
        #. 反向归一化操作        
        Ynew2[:,j] = Ynew2[:,j]*maxValue[j]
    RegionFurture[i] = Ynew2
    # nY = Ynew2.detach().cpu().numpy()
    