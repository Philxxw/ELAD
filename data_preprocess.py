# %load data_preprocess.py
import ast
import csv
import os
import sys
from pickle import dump
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from tfsnippet.utils import makedirs

#MSL 55;SMAP 25;SMD 38
kpi_number = 55

def txt_to_csv(category, filename, dataset,dataset_folder):
    file = os.path.join(output_folder, dataset + "_" + category + ".csv")
    with open(file, 'w+', newline='', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')  # dialect可用delimiter= ','代替，后面的值是自己想要的符号
        with open(os.path.join(dataset_folder, category, filename), 'r', encoding='utf-8') as filein:
            for line in filein:
                line_list = line.strip('\n').split(',')  #
                spamwriter.writerow(line_list)
    data = pd.read_csv(file, header=None)
    df = MinMaxScaler().fit_transform(data)
    data_csv = pd.DataFrame(df)
#     df_0_7 = data_csv.iloc[:,0:8]
#     print(df_0_7.shape)
#     data_csv = pd.concat([data_csv,df_0_7],axis = 1)
#     print(data_csv.shape)
    data_csv.to_csv(file,index = None,header = None)
    return data_csv


def token_input(data, category, dataset, embedding_size=32, h_number=5, kpi_number=kpi_number):
    line_number = len(data) - embedding_size * h_number + 1
    data_rand = pd.DataFrame(np.zeros([line_number, 1]))
    data_rand = pd.DataFrame(np.array(range(line_number)))
    data_rand = data_rand * kpi_number
    # np.tile(a,(2,1))第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数
    data_rand_38 = pd.DataFrame(np.tile(data_rand, [1, kpi_number]))
    # data_rand_38:
    # [0*38,0*38,...,*38]
    # [1*38,1*38...,1*38]
    # ...
    # [(line_number-1)*38,...]]
    data_32_mul_38 = pd.DataFrame(np.tile(np.array([embedding_size * kpi_number]), [line_number, kpi_number]))
    for h_index in range(0, h_number):
        if h_index < 1:
            data_rand_38_all = data_rand_38
        else:
            data_rand_h = data_rand_38 + data_32_mul_38 * h_index
            data_rand_38_all = pd.concat([data_rand_38_all, data_rand_h], axis=1)
            # print(data_rand_38_all.iloc[0, :])
    data_0to37 = pd.DataFrame(np.array(range(kpi_number))).T
    data_0to37_all = pd.DataFrame(np.tile(data_0to37, [line_number, h_number]), columns=data_rand_38_all.columns)
    # print(data_0to37_all)
    # datra_0to37_all:
    # [[0,1...37]
    # [0,1...37]
    # ...
    # [0,1...37]]
    data_rand_38_all = data_rand_38_all + data_0to37_all
    if category == 'test':
        data_rand_38_all.to_csv(os.path.join(output_folder, dataset + "_" + category + "_" + "tocken_input.csv"),
                                header=False, index=None)
    elif category == 'train':
        #data_rand_38_all = data_rand_38_all.reindex(np.random.permutation(data_rand_38_all.index))
        data_rand_38_all.to_csv(os.path.join(output_folder, dataset + "_" + category + "_" + "tocken_input.csv"),
                                header=False, index=None)
    else:
        print("input error!")

def embedding(data, category, dataset, embedding_size=32, kpi_number=kpi_number):
    vocab_size = (len(data) - embedding_size + 1) * kpi_number + 1
    embedding_dict = pd.DataFrame(np.ones([vocab_size, embedding_size]))
    for dict_index in range(len(data) - embedding_size + 1):
        #    dict_index = 2
        # if dict_index % 1000 == 0:
        #     print(dict_index)
        data_mid = data.iloc[dict_index:dict_index + embedding_size, :]
        data_mid = data_mid.T
        embedding_dict.iloc[dict_index * kpi_number:(dict_index + 1) * kpi_number, :] = np.array(data_mid)

    embedding_dict.to_csv(os.path.join(output_folder, dataset + "_" + category + "_" + "embedding_dict.csv"), header=None, index=None)


if __name__ == '__main__':
    
    dataset_folder = 'SMAP_and_MSL'
    #MSL;SMAP #'MSL_and_SMAP'
    #SMD #'ServerMachineDataset'
    
    output_folder = 'prepare_input_MSL'
    #'prepare_input_SMD'#'prepare_input_SMAP'
    os.makedirs(output_folder, exist_ok=True)
    #file_list = os.listdir(os.path.join(dataset_folder, "train"))
    file_list = ['MSL.txt']
    
    for filename in file_list:
        data_train = txt_to_csv('train', filename, filename.strip('.txt'), dataset_folder)
        data_test = txt_to_csv('test', filename, filename.strip('.txt'), dataset_folder)
        data_label = txt_to_csv('test_label', filename, filename.strip('.txt'), dataset_folder)
#         data_train = pd.read_csv(os.path.join(dataset_folder, 'train', filename),header = None)
#         data_test = pd.read_csv(os.path.join(dataset_folder, 'test', filename),header = None)
        token_input(data_train, 'train', filename.strip('.txt'))
        token_input(data_test, 'test', filename.strip('.txt'))

        embedding(data_train, 'train', filename.strip('.txt'))
        embedding(data_test, 'test', filename.strip('.txt'))

