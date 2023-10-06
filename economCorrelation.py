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
from minepy import MINE
from matplotlib import font_manager
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10)


class CyrusPlot(object):
    def __init__(self,dpi=72,fig_size=[30,20]):
        """
        实列化该类，然后直接调用cyrus_heat_map方法
        :param dpi:
        :param fig_size:
        """
        self.dpi = dpi
        self.fig_size = fig_size
        self.font = font_manager.FontProperties(fname="C:\Windows\Fonts\simsun.ttc", size=10)

    def cyrus_heat_map(self,datas,x_ticks = [],y_ticks = [],bar_label = "bar label",show = True,save_name = ""):
        figure = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        ax = figure.add_subplot(111)
        if not x_ticks:
            x_ticks = ["x"+str(i) for i in range(datas.shape[1])]
            y_ticks = ["y" + str(i) for i in range(datas.shape[0])]
        im, _ = self.heatmap(np.array(datas), x_ticks, y_ticks,
                        cmap="Greens", cbarlabel=bar_label,ax=ax)  # plt.cm.RdBu   PuOr Oranges
        """['Accent','Accent_r','Blues','Blues_r','BrBG','BrBG_r','BuGn','BuGn_r','BuPu','BuPu_r','CMRmap','CMRmap_r',\
            'Dark2','Dark2_r','GnBu','GnBu_r','Greens','Greens_r','Greys','Greys_r','OrRd','OrRd_r','Oranges','Oranges_r',\
                'PRGn','PRGn_r','Paired','Paired_r','Pastel1','Pastel1_r'
        """        
        self.annotate_heatmap(im, valfmt="{x:.2f}", size=16)
        if save_name:
            plt.savefig("./figure/" + save_name + ".jpg")
        if show:
            plt.show()

    def heatmap(self,data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        if not ax:
            ax = plt.gca()
        im = ax.imshow(data, **kwargs)
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontproperties=font_manager.FontProperties(fname="C:\Windows\Fonts\simsun.ttc", size=8))
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(col_labels,fontproperties=self.font)
        ax.set_yticklabels(row_labels,fontproperties=self.font)
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def annotate_heatmap(self,im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw):

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        kw = dict(horizontalalignment="center",
                  verticalalignment="center",
                  )
        kw.update(textkw)

        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[abs(data[i, j]) > 0.5])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

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
Fdata = np.zeros(shape=(31,27,18))
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
    Fdata[i,:,:] = fdata2
    
Xdata = fdataT.reshape(Xdata.shape)  
maxValue = 1.1 * np.max(Xdata,0)

## 归一化数据
Xdata2 = np.zeros(Xdata.shape)
for i in range(27):
    Xdata2[:,i] = Xdata[:,i]/maxValue[i] 


Corr = np.zeros((31,31,27,27))

for i in range(31):
    for j in range(31):
        for k in range(27):
            for s in range(27):
                mine = MINE(alpha=0.6, c=15)
                mine.compute_score(Fdata[i,k,:],Fdata[j,s,:])
                Corr[i,j,k,s] = mine.mic()
plot_tool = CyrusPlot()
plot_tool.cyrus_heat_map(np.power(Corr[10,25],5),show=True)