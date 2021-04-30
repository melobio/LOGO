# !/usr/bin/env python3
# encoding: utf-8
"""
@author: Licko, huanglichao
@time: 2020/12/2 9:50
@desc:

分别对919/2002/3357进行HGMD logistic 训练与预测
1、读取HGMD 7188的919/2002/3357染色质特征（由GBERT-919/2002/3357模型预测得到）
2、数据处理，提取每一子集的对应染色质特征，处理为DMatriX，用于后续logistic训练
3、完成训练、验证，在测试集上计算效果，返回AUROC，并将预测概率值与变异信息一并输出
4、获取DeepSEA的三个logistic回归模型对测试集变异的预测结果

############ Part1: Getting chromatin feature from {} file ############
############ Part2: Getting DMatrix with {} marks ############
############ Part3: Training Xgboost model with {} marks ############
############ Part4: Evaluating with {} marks ############
############ Part5: Saving Xgboost model in {} marks ############
############ Part6: Getting deepsea 3 logistic output value ############

"""
import os
import h5py
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score


# VCF_FILE
# CHROMATIN_df

def get_rs_name(VCF_FILE):
    """
    :param VCF_FILE:输入的VCF文件，需要保证第三列为name，如HGMD12
    :return: 返回第三列name信息
    """
    # 取子集name
    rs_name_list = []
    with open(VCF_FILE, 'r') as rd:
        for ii in rd.readlines():
            rs_name_list.append(ii.split('\t')[2])
            # print(ii.split('\t')[2])
    rd.close()
    print(len(rs_name_list))
    return rs_name_list


def get_snpclass_value(VCF_FILE, CHROMATIN_df):
    """
    :param VCF_FILE:     输入的VCF文件，需要保证第三列为name，如HGMD12
    :param CHROMATIN_df: 包含7188各变异的所有919/2002/3357特征的dataframe文件
    :return: 返回根据给定list提取出来的变异染色质特征文件子集，如从7188从提取出719测试集对应的染色质特征
    """
    # 取子集name
    rs_name_list = get_rs_name(VCF_FILE)

    # 根据vcf的变异，提取子集，同时依据rs_name顺序，保证提取的特征的顺序与vcf一致，便于比较
    index_list = []
    for i_n in rs_name_list:
        # print(new_df[new_df.name == i_n][:, :5])
        tem_df = CHROMATIN_df[CHROMATIN_df.name == i_n]
        if tem_df.shape[0] != 0:
            index_list.append(tem_df.index.tolist()[0])
    sub_df_ori = CHROMATIN_df.filter(items=index_list, axis=0)
    print('Chromatin feature df shape: ', sub_df_ori.shape)

    return sub_df_ori


def get_DMatrix(VCF_FILE, CHROMATIN_df):
    """
    :param VCF_FILE:     输入的VCF文件，需要保证第三列为name，如HGMD12
    :param CHROMATIN_df: 包含7188各变异的所有919/2002/3357特征的dataframe文件
    :return: 返回染色质特征子集，同时处理为DMatrix格式，返回DMatrix
    """
    # 取子集name
    rs_name_list = get_rs_name(VCF_FILE)

    # 根据vcf的变异，提取子集，同时依据rs_name顺序，保证提取的特征的顺序与vcf一致，便于比较
    index_list = []
    for i_n in rs_name_list:
        # print(new_df[new_df.name == i_n][:, :5])
        tem_df = CHROMATIN_df[CHROMATIN_df.name == i_n]
        if tem_df.shape[0] != 0:
            index_list.append(tem_df.index.tolist()[0])
    sub_df_ori = CHROMATIN_df.filter(items=index_list, axis=0)
    print('Chromatin feature df shape: ', sub_df_ori.shape)

    # 添加label
    sub_df_ori['label'] = np.where(sub_df_ori.name.str.startswith('1000G'), 0, 1)
    # 检查是否有缺失值
    if len(sub_df_ori[sub_df_ori.isnull().values == True]) != 0:
        print('Have NA!')
    # 特征提取与label处理
    sub_df = sub_df_ori.iloc[:, 5:]
    y = np.asarray(sub_df['label'])
    X = np.asarray(sub_df.drop(['label'], axis=1))
    print("X.shape: {}, y.shape: {}".format(X.shape, y.shape))

    # 标准化处理
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    std_X = ss.fit_transform(X)
    print('after standard mean and std is ', np.mean(std_X), np.std(std_X))

    # xgb预处理
    DM_Xy = xgb.DMatrix(std_X, y)

    return DM_Xy, std_X, X, y, sub_df_ori

"""
PART1
Main
"""

# 输入染色质差异所在文件
mark_type_list = [919, 2002, 3357]

for mark_type in mark_type_list:

    print('############ Part1: Getting chromatin feature from {} file ############'.format(mark_type))
    MAIN_PATH = '/alldata/Nzhang_data/project/T2D/1.GWAS_LD_predict/5.softwarepred/train/fst0.01/posstrand_result/GeneBert/'
    CHROMATIN_FILE = MAIN_PATH + \
                     '1000G_HGMD_posstrand_5.vcf_128bs_5gram_{}feature.out.logfoldchange.csv'.format(mark_type)
    CHROMATIN_df = pd.read_csv(CHROMATIN_FILE)  # CHROMATIN_df.head()
    VCF_PATH = '/alldata/LChuang_data/myP/GeneBert/BGI-Gene_new/examples/CADD/HGMD/'

    # vcf
    for ii in range(10):
        task_list = ['train', 'valid', 'test']
        outputpath_list = []
        for task in task_list:
            # VCF_FILE = VCF_PATH + \
            # '1kb_HGMD_7188/shuffle_10time_6ngram/1000G_HGMD_posstrand_8softwares_5_{}_shuffle8.vcf'.format(task)
            VCF_FILE = VCF_PATH + \
                       '1kb_HGMD_7188/shuffle_10time_6ngram/1000G_HGMD_posstrand_8softwares_5_{}_shuffle{}.vcf'.format(
                           task, ii)
            print(VCF_FILE)
            outputpath_list.append(VCF_FILE)

        print('############ Part2: Getting DMatrix with {} marks ############'.format(mark_type))
        DM_Xy_train, std_X_train, X_train, y_train, _ = get_DMatrix(outputpath_list[0], CHROMATIN_df)
        DM_Xy_valid, std_X_valid, X_valid, y_valid, _ = get_DMatrix(outputpath_list[1], CHROMATIN_df)
        DM_Xy_test, std_X_test, X_test, y_test, test_df = get_DMatrix(outputpath_list[2], CHROMATIN_df)

        # Training
        print('############ Part3: Training Xgboost model with {} marks ############'.format(mark_type))
        num_round = 1000  # 迭代次数=100
        l2 = 2000
        l1 = 20
        threads = 40
        param = {
                 # 'max_depth':8,
                 'booster': 'gbtree',  # booster': 'gblinear',
                 'alpha': l1,
                 'lambda': l2,
                 'eta': 0.1,
                 'objective':'binary:logistic',  # 'objective': 'reg:squarederror',
                 'nthread': threads,
                 'eval_metric': 'auc',
                 'verbosity': 0}

        evallist = [(DM_Xy_train, 'train'), (DM_Xy_valid, 'eval')]  # 评估性能过程，有前后顺序要求
        raw_model = xgb.train(params=param,
                              dtrain=DM_Xy_train,
                              num_boost_round=num_round,
                              evals=evallist,
                              verbose_eval=False,        # 不显示eval过程
                              # early_stopping_rounds=100  # 该参数无法更好体现出模型在test上的性能
                             )

        # Predicting
        print('############ Part4: Evaluating with {} marks ############'.format(mark_type))
        pred_test_raw = raw_model.predict(DM_Xy_test)
        pred_test_acc = raw_model.predict(DM_Xy_test)
        for i in range(len(pred_test_raw)):
            if pred_test_acc[i] > 0.5:
                pred_test_acc[i] = 1
            else:
                pred_test_acc[i] = 0
        print('acc:', accuracy_score(DM_Xy_test.get_label(), pred_test_acc))
        print('AUROC:', roc_auc_score(DM_Xy_test.get_label(), pred_test_raw))

        # Saving
        print('############ Part5: Saving Xgboost model in {} marks ############'.format(mark_type))
        test_df = test_df.iloc[:, :5]
        test_df['value'] = pred_test_raw
        basename = os.path.basename(VCF_FILE)
        out = '/alldata/LChuang_data/myP/GeneBert/HGMD_logistic/' + basename.replace('.vcf',
                                                                                     '_XGboost_{}mark.predict'.format(
                                                                                         mark_type))
        out_model = '/alldata/LChuang_data/myP/GeneBert/HGMD_logistic/' + basename.replace('.vcf',
                                                                                           '_XGboost_{}mark.model'.format(
                                                                                               mark_type))
        raw_model.save_model(out_model)
        print('Saving model: \n', out_model)
        # bst_new = xgb.Booster({'nthread':4}) #init model
        # bst_new.load_model("HGMD_logistic/0001.model") # load data

        test_df.to_csv(out, sep='\t', index=None, header=None)
        print('Saving prediction file: \n', out)


"""
PART2
"""
# Getting deepsea 3 logistic output value
for ii in range(10):
    VCF_PATH = '/alldata/LChuang_data/myP/GeneBert/BGI-Gene_new/examples/CADD/HGMD/'
    VCF_FILE = VCF_PATH + \
               '1kb_HGMD_7188/shuffle_10time_6ngram/1000G_HGMD_posstrand_8softwares_5_test_shuffle{}.vcf'.format(ii)
    basename = os.path.basename(VCF_FILE)
    # Getting deepsea 3 logistic output value
    print("############ Part6: Getting deepsea 3 logistic output value ############")
    snpclass = '/alldata/Nzhang_data/project/T2D/1.GWAS_LD_predict/5.softwarepred/train/fst0.01/posstrand_result/DeepSEA/infile.vcf.out.snpclass'
    sss_df = pd.read_csv(snpclass)
    # 去除第一列
    sss_df = sss_df.iloc[:, 1:]
    snpcalss_outputpath = '/alldata/LChuang_data/myP/GeneBert/HGMD_logistic/' + \
                          basename.replace('.vcf', '_DeepSEA_snpclass.snpclass')
    sub_snpclass_df_ori = get_snpclass_value(VCF_FILE, sss_df)
    sub_snpclass_df_ori.to_csv(snpcalss_outputpath, sep='\t', index=None)
    print('Saving snpclass file: \n', snpcalss_outputpath)