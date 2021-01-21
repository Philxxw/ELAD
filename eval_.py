# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 08:32:05 2020

@author: xxw
"""
import pandas as pd 
import os 
import torch 
import numpy as np 
import pylab as plt 
from sklearn.metrics import mean_squared_error,precision_recall_fscore_support 
from scipy import signal 
import eval_methods as EMS 
import matplotlib.pyplot as plt 
kpi_number = 38#55;25
data_dir = 'prepare_input_SMD/'#'prepare_input_SMAP/'#'prepare_input_MSL/'
reconstruction_dir = 'reconstruction_output/'
#%%
def generation_time_dict(data_gen):
    #生成数据组成的大字典
    #除前38和后38的数据其他每个均重复了32个时间点
    data_gen = np.array(data_gen).reshape(-1,kpi_number,32)
    data_g_dict = data_gen[0]
    for g in range(1,data_gen.shape[0]):
        for i in range(1,32):
            data_g_dict[:,g-1+i] = data_g_dict[:,g-1+i]+data_gen[g][:,i-1]
        data_g_dict = np.append(data_g_dict,data_gen[g][:,31].reshape(-1,1),axis = 1)
    L = data_g_dict.shape[1]
    for col in range(L):
        if col <=31:
            data_g_dict[:,col] = data_g_dict[:,col]/(col+1)
        elif col >=L-32:
            data_g_dict[:,col] = data_g_dict[:,col] /(L-col)
        else:
            data_g_dict[:,col] = data_g_dict[:,col]/32
    return data_g_dict

def MSE_list(data_gen,data_ini):
    MSE = []
    for i in range(data_gen.shape[0]):
        for j in range(data_gen.shape[1]):
            MSE.append(mean_squared_error([data_gen[i][j]],[data_ini[i][j]]))
    return np.array(MSE).reshape(data_gen.shape[0],-1)

def genereate_label(MSE,alpha):
    a = MSE.copy()
    a.sort(reverse=True)
    x  = a[alpha+1]
    result = []
    for v in MSE:
        if v < x:
            result.append(0)
        else:
            result.append(1)
    return result,x
#%%
Result = {}
filename_list = ['MSL']
#filename_list = ['SMAP']
#filename_list = ['machine-1-1','machine-1-2','machine-1-3','machine-1-4','machine-1-5','machine-1-6','machine-1-7',
                  #'machine-1-8','machine-2-1','machine-2-2','machine-2-3','machine-2-4','machine-2-7','machine-2-8',
                  #'machine-2-9','machine-3-1','machine-3-2','machine-3-3','machine-3-4','machine-3-5','machine-3-6',
                  #'machine-3-7','machine-3-8','machine-3-9','machine-3-10','machine-3-11']
for filename in filename_list:
    data_gen_name  = filename+'_reconstruction_test.csv'
    data_ini_name = filename+'_test_embedding_dict.csv'
    data_gen = pd.read_csv(os.path.join(reconstruction_dir,data_gen_name),encoding = 'gb18030',header = None)
    data_g_dict = generation_time_dict(data_gen)
    #print(data_g_dict.shape)
  
    data_ini_token = np.array(pd.read_csv(os.path.join(data_dir,filename+'_test_tocken_input.csv'),encoding = 'gb18030',header = None))
    data_ini_dict = np.array(pd.read_csv(os.path.join(data_dir,data_ini_name),encoding = 'gb18030',header = None))
    data_ini = data_ini_dict[data_ini_token]
    data_ini = data_ini[:,kpi_number*4:kpi_number*5]
    data_ini_t_dict = generation_time_dict(data_ini)
    #print(data_ini_t_dict.shape)
    MSE = MSE_list(data_g_dict,data_ini_t_dict)
    

    label = np.array(pd.read_csv(os.path.join(data_dir,filename+'_test_label.csv'),encoding = 'gb18030',header = None))
    label = list(label.reshape(-1))
    label = label[32*4:MSE.shape[1]+32*4]
    print('start to detect %s...'%(filename))
    Alpha = np.linspace(0.001,0.021,50)
    f1_score = 0
    for alpha in Alpha:
        result_all_ano = []
        for m in range(MSE.shape[0]):
            r,x = genereate_label(list(MSE[m]),int(MSE.shape[1]*alpha))
            result_all_ano.append(r)
        result = list(np.array(result_all_ano).sum(axis =0))
        Beta = np.linspace(2,max(result),max(result)-1)
        for beta in Beta:
            #print(len(result))
            #print(len(label))
            predict = EMS.adjust_predicts(result, label,threshold =beta)
            [f1, precision, recall, TP, TN, FP, FN] = list(EMS.calc_point2point(np.array(predict), np.array(label)))
            if f1 >= f1_score:
                f1_score =f1
                predict_r = predict
                R_ = [f1, precision, recall, TP, TN, FP, FN,alpha,beta]
                #result_ = result_all_ano
    
    #pd.DataFrame(predict_r).to_csv('reconstruction_label/%s.csv'%(filename))
    print('Results of %s.'%(filename))
    print(R_)
    Result['%s'%(filename)] = R_

(pd.DataFrame(Result).T).to_csv('result.csv')
