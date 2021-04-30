# -*- coding:utf-8 -*-

import pickle
import os
import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from multiprocessing import Pool


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_word_dict_5gram(word_index_from=3):
    word_dict = {}
    N = [0, 1, 2, 3, 4]

    index = word_index_from
    for n in N:
        # print(n)
        A = N
        for a in A:
            # print(n+a)
            G = N
            for g in G:
                # print(n+a+g)
                C = N
                for c in C:
                    # print(n+a+g+c)
                    T = N
                    for t in T:
                        # print(n+a+g+c+t)
                        # dict_list.append(n+a+g+c+t)
                        # print(int(n * 10**4 + a * 10**3 + g * 10**2 + c * 10**1 + t), (n + a + g + c + t))
                        word_dict[int(n * 10**4 + a * 10**3 + g * 10**2 + c * 10**1 + t)] = index
                        index += 1
    return word_dict




def get_word_dict_6gram(word_index_from=3):
    N = [0, 1, 2, 3, 4]

    word_dict = {}
    N = [0, 1, 2, 3, 4]
    # 从3开始
    index = word_index_from
    for n in N:
        # print(n)
        A = N
        for a in A:
            # print(n+a)
            G = N
            for g in G:
                # print(n+a+g)
                C = N
                for c in C:
                    # print(n+a+g+c)
                    T = N
                    for t in T:
                        # print(n+a+g+c+t)
                        # dict_list.append(n+a+g+c+t)
                        S = N
                        for s in S:
                            # print(n+a+g+c+t+s)
                            # dict_list.append(n+a+g+c+t+s)
                            # word_dict[n + a + g + c + t + s] = index
                            word_dict[int(n * 10 ** 5 + a * 10 ** 4 + g * 10 ** 3 + c * 10 ** 2 + t * 10 ** 1 + s)] = index
                            index += 1
    return word_dict


def get_word_dict_for_n_gram(word_index_from=3, n_gram:int=6, alphabet:list=['A','C','T','G','U'], predefined_tokens:list=[]):
    word_dict = {}

    if predefined_tokens is not None and len(predefined_tokens) > 0:
        for token in predefined_tokens:
            word_dict[token] = len(word_dict)

    word_set = set()
    previous_layer_word_set = set()

    word_index_from = max(word_index_from, len(predefined_tokens))
    for ii in range(n_gram):
        for word in alphabet:
            if ii == 0:
                word_set.add(word)
                word_dict[word] = word_index_from + len(word_dict)
            else:
                for add_word in previous_layer_word_set:
                    if len(add_word) == ii:
                        new_word = add_word + word
                        word_set.add(new_word)
                        word_dict[new_word] = word_index_from + len(word_dict)
        previous_layer_word_set = word_set
        word_set = set()
    return word_dict


def get_word_dict_for_n_gram_number(word_index_from=3, n_gram:int=6, alphabet:list=[0, 1, 2, 3, 4], predefined_tokens:list=[]):
    word_dict = {}

    if predefined_tokens is not None and len(predefined_tokens) > 0:
        for token in predefined_tokens:
            word_dict[token] = len(word_dict)

    word_set = set()
    previous_layer_word_set = set()
    add_word_set = set()
    word_index_from = max(word_index_from, len(predefined_tokens))
    for ii in range(n_gram):
        for word in alphabet:
            if ii == 0:
                word_set.add(word)
                add_word_set.add(word)
                word_dict[word] = word_index_from + len(word_dict)
            else:
                for add_word in previous_layer_word_set:
                    if len(str(add_word)) == ii:
                        new_word = add_word * 10 + word
                        if new_word in add_word_set:
                            continue
                        #print("new_word: ", new_word, word_set)
                        word_set.add(new_word)
                        add_word_set.add(new_word)
                        word_dict[new_word] = word_index_from + len(word_dict)
        previous_layer_word_set = word_set
        word_set = set()
    return word_dict



if __name__ == '__main__':
    prefix = "/data/HPhuang_data/DeepSEA/"
    prefix = "F:\\Research\\Data\\DeepSEA\\deepsea_train\\"
    train_path = prefix + 'train.mat'
    valid_path = prefix + 'valid.mat'
    test_path = prefix + 'test.mat'

    dict_path = "../../data/word_dict_5gram.pkl"

    train_data_path = 'F:\\Research\\Data\\DeepSEA\\deepsea_train\\train_3gram_number\\'

    word_index_from = 3

    n_gram = 3
    stride = 3

    number_dict = get_word_dict_for_n_gram_number(n_gram=n_gram)
    print(number_dict)

    n_gram_value = np.ones(n_gram)
    for ii in range(n_gram):
        n_gram_value[ii] = int(n_gram_value[ii] * (10 ** (n_gram - ii - 1)))
        print(10 ** (n_gram - ii - 1))
    print("n_gram_value: ", n_gram_value)

    # 读取训练文件
    files = os.listdir(train_data_path)
    for file_name in files:
        print("File: ", file_name)
        loaded = np.load(os.path.join(train_data_path, file_name))
        x_train = loaded['x']
        y_train = loaded['y']

        # max_seq_len = x_train.shape[0]
        #
        # max_slice_seq_len = x_train.shape[1] // n_gram * n_gram
        #
        # print("max_slice_seq_len: ", max_slice_seq_len)
        #
        # slice_indexes = []
        # kk = 2
        #
        # for gg in range(0, max_slice_seq_len, 1):
        #     slice_indexes.append(gg)
        #
        # print("slice_indexes: ", slice_indexes)
        # for ii in slice_indexes:
        #     #for jj in range(n_gram):
        #         #print(x_train[:, jj:(jj+n_gram)], x_train[:, jj:(jj+n_gram)].shape)
        #     print(ii, (ii + n_gram))
        #     result = np.dot(x_train[:, ii:(ii + n_gram - 1)], n_gram_value)
        #     result = np.array(result, dtype=np.int)
        #         #print()

        # print("slice_indexes: ", slice_indexes)
        # print("slice_indexes: ", len(slice_indexes))

        # x_train = x_train - word_index_from
        print("Max: ", np.max(x_train))
        print("Min: ", np.min(x_train))

        print("Max: ", np.max(y_train))
        print("Min: ", np.min(y_train))
        break
