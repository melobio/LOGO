import os, sys
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

"""
00_DataPrepare.py
"""

sys.path.append("../")
from bgi.bert4keras.models import build_transformer_model
from bgi.common.callbacks import LRSchedulerPerStep
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number


def preccess_data(seq_data: np.ndarray,
                  actg_value,
                  stride,
                  seq_max_len,
                  n_gram,
                  n_gram_value,
                  num_word_dict,
                  ):
    """
    分批处理数据，生成npz文件
    :param seq_data:
    :param actg_value:
    :param step:
    :param n_gram:
    :param n_gram_value:
    :param num_word_dict:
    :return:
    """

    # AGCT转换为1，2，3，4
    actg = np.matmul(seq_data, actg_value)
    gene = []
    for kk in range(0, len(actg), stride):
        if kk >= seq_max_len:
            break

        actg_temp_value = 0
        if kk + n_gram <= len(actg):
            actg_temp_value = np.dot(actg[kk:kk + n_gram], n_gram_value)
            actg_temp_value = int(actg_temp_value)
        else:
            for gg in range(kk, len(actg)):
                actg_temp_value += actg[gg] * (10 ** (n_gram - gg % n_gram - 1))

            # print("10 ** (kk % n_gram): ", 10 ** (kk % n_gram))
            actg_temp_value = actg_temp_value * (10 ** (kk % n_gram))

        gene.append(num_word_dict.get(actg_temp_value, 0))
    # print("gene: ", len(gene), seq_max_len, len(actg))
    return np.array(gene)


def load_npz_data(data_path, prefix='', ngram=3, reshape=True, NUM_SEQ=4, masked=True):
    """
    导入npz数据
    :param file_name:
    :param ngram:
    :param only_one_slice:
    :param ngram_index:
    :return:
    """
    x_data_all = []
    y_data_all = []
    file_names = []

    actg_value = np.array([1, 2, 3, 4])
    n_gram_value = np.ones(ngram)
    for ii in range(ngram):
        n_gram_value[ii] = int(n_gram_value[ii] * (10 ** (ngram - ii - 1)))
    print("n_gram_value: ", n_gram_value)

    num_word_dict = get_word_dict_for_n_gram_number(n_gram=ngram)

    files = os.listdir(data_path)
    for file_name in files:
        print(file_name)
        if str(file_name).startswith(prefix) and str(file_name).endswith('.npz'):
            file_names.append(os.path.join(data_path, file_name))

            loaded = np.load(os.path.join(data_path, file_name), allow_pickle=True)
            x_data = loaded['sequence']
            y_data = loaded['label']

            print("x_data: ", x_data.shape)
            print("y_data: ", y_data.shape)

            x_data_all.append(x_data)
            y_data_all.append(y_data)
            # break

    x_data_all = np.vstack(x_data_all)
    print("x_data_all: ", x_data_all.shape)
    x_gene_seq_all = []
    if reshape is True:
        x_data_all = np.reshape(x_data_all, (x_data_all.shape[0], x_data_all.shape[1] // NUM_SEQ, NUM_SEQ))
        seq_max_len = x_data_all.shape[1] // ngram // ngram * ngram * ngram

        for ii in range(len(x_data_all)):
            seq_data = x_data_all[ii]
            gene_seq = preccess_data(seq_data,
                                     actg_value=actg_value,
                                     stride=1,
                                     seq_max_len=seq_max_len,
                                     n_gram=ngram,
                                     n_gram_value=n_gram_value,
                                     num_word_dict=num_word_dict,
                                     )
            x_gene_seq_all.append(gene_seq)

            if ii > 0 and ii % 10000 == 0:
                print(ii)
        x_data_all = np.array(x_gene_seq_all)

    y_data_all = np.hstack(y_data_all)

    return x_data_all, y_data_all


def main():
    CELL = 'tB'
    TYPE = 'P-E'

    ngram = 6


    task = 'test'
    ## load data: sequence
    if task == 'train':
        seq_data_path = CELL + '/' + TYPE + '/{}_gram/'.format(str(ngram))
        if os.path.exists(seq_data_path) is False:
            os.makedirs(seq_data_path)

        data_path = CELL + '/' + TYPE + '/'
        region1_seq, label = load_npz_data(data_path, ngram=ngram, prefix='enhancer_Seq')
        print("region1_seq: ", region1_seq.shape)
        np.savez(os.path.join(seq_data_path, 'enhancer_Seq_{}_gram.npz'.format(ngram)), x=region1_seq, y=label)

        region1_seq, label = load_npz_data(data_path, ngram=ngram, prefix='promoter_Seq')
        np.savez(os.path.join(seq_data_path, 'promoter_Seq_{}_gram.npz'.format(ngram)), x=region1_seq, y=label)
    else:
        seq_data_path = CELL + '/' + TYPE + '/test/{}_gram/'.format(str(ngram))
        if os.path.exists(seq_data_path) is False:
            os.makedirs(seq_data_path)

        data_path = CELL + '/' + TYPE + '/test/'
        region1_seq, label = load_npz_data(data_path, ngram=ngram, prefix='enhancer_Seq')
        print("region1_seq: ", region1_seq.shape)
        np.savez(os.path.join(seq_data_path, 'enhancer_Seq_{}_gram.npz'.format(ngram)), x=region1_seq, y=label)

        region1_seq, label = load_npz_data(data_path, ngram=ngram, prefix='promoter_Seq')
        np.savez(os.path.join(seq_data_path, 'promoter_Seq_{}_gram.npz'.format(ngram)), x=region1_seq, y=label)






"""RUN"""
main()
