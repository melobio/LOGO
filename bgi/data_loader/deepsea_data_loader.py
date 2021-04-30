
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import h5py

word_dict = {}


def load_data(train_path='train.mat', valid_path='valid.mat', test_path='test.mat', index_from=3, kernel=5, shuffle=True, **kwargs):
    """
    """
    word_dict.clear()

    x_train = []
    y_train = []
    train_counter = 1
    if train_path is not None:
        loaded = h5py.File(train_path, 'r')
        data = loaded['trainxdata']
        labels = loaded['traindata']

        print("Data: ", data.shape)  # (1000, 4, 4400000)
        print("Labels: ", labels.shape)  # (919, 4400000)

        indexes = np.arange(data.shape[2])
        if shuffle == True:
            np.random.shuffle(indexes)

        index = 0
        for ii in range(data.shape[2]):
            actg = np.zeros(data.shape[0])
            # 进行ACGT的转换,0，1，2，3
            for jj in range(data.shape[1]):
                actg = actg + data[:,jj,ii] * (10 ** jj)  # ACGT，0，1，2，3对应0,10,100,1000

            # 将ACGT转为1234
            actg = np.log10(actg) + 1
            actg = np.array(actg, dtype=int)
            # 每5个单位作为一个间隔，每5bp为一个片段
            gene = []
            for jj in range(0, len(actg), kernel):
                gene_str = ''
                for kk in range(kernel):
                    gene_str += str(actg[jj+kk])
                # 将遇到的新item给定编号并加入word_dict中，编号从3开始
                if gene_str not in word_dict:
                    word_dict[gene_str] = len(word_dict) + index_from
                gene.append(word_dict[gene_str])
            #print("gene: ", gene)
            x_train.append(np.array(gene))
            y_train.append(labels[:, ii])

            if index % 10000 == 0 and index > 0:
                print(index)
                #break
            if index % 100000 == 0 and index > 0:
                x_train = np.array(x_train)
                y_train = np.array(y_train)
                save_dict = {
                    'x': x_train,
                    'y': y_train
                }
                save_file = train_path.replace('.mat', '_ngram_5_{}.npz'.format(train_counter))
                np.savez(save_file, **save_dict)
                x_train = []
                y_train = []
                train_counter += 1
            index += 1

            # if index  > 100000 * 2:
            #     break
            #print(index)

    if len(x_train) > 0:
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        save_dict = {
            'x': x_train,
            'y': y_train
        }
        save_file = train_path.replace('.mat','_{}.npz'.format(train_counter))
        np.savez(save_file, **save_dict)

    x_test = []
    y_test = []
    test_counter = 1
    if test_path is not None:

        loaded = sio.loadmat(test_path)
        data = loaded['testxdata']
        labels = loaded['testdata']

        index = 0
        for ii in range(data.shape[0]):
            actg = np.zeros(data.shape[2])
            for jj in range(data.shape[1]):
                actg = actg + data[ii][jj] * (10 ** jj)

            actg = np.log10(actg) + 1
            actg = np.array(actg, dtype=int)

            gene = []
            for jj in range(0, len(actg), kernel):
                gene_str = ''
                for kk in range(kernel):
                    gene_str += str(actg[jj + kk])
                if gene_str not in word_dict:
                    word_dict[gene_str] = len(word_dict) + index_from
                gene.append(word_dict[gene_str])
            # print("gene: ", gene)
            x_test.append(np.array(gene))
            y_test.append(labels[ii])

            if index % 10000 == 0 and index > 0:
                print(index)
                # break
            if index % 100000 == 0 and index > 0:
                x_test = np.array(x_test)
                y_test = np.array(y_test)
                save_dict = {
                    'x': x_test,
                    'y': y_test
                }
                save_file = test_path.replace('.mat', '_ngram_{}_{}.npz'.format(kernel, test_counter))
                np.savez(save_file, **save_dict)
                x_test = []
                y_test = []
                test_counter += 1
            index += 1

            # if index > 100000 * 2:
            #     break
            # print(index)

    if len(x_test) > 0:
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        save_dict = {
            'x': x_test,
            'y': y_test
        }
        save_file = test_path.replace('.mat', '_ngram_{}_{}.npz'.format(kernel, test_counter))
        np.savez(save_file, **save_dict)


    x_valid = []
    y_valid = []
    if valid_path is not None:
        loaded = sio.loadmat(valid_path)
        data = loaded['validxdata']
        labels = loaded['validdata']

        for ii in range(data.shape[0]):
            actg = np.zeros(data.shape[2])
            for jj in range(data.shape[1]):
                actg = actg + data[ii][jj] * (10 ** jj)

            actg = np.log10(actg) + 1
            actg = np.array(actg, dtype=int)

            gene = []
            for jj in range(0, len(actg), kernel):
                gene_str = ''
                for kk in range(kernel):
                    gene_str += str(actg[jj + kk])
                if gene_str not in word_dict:
                    word_dict[gene_str] = len(word_dict) + index_from
                gene.append(word_dict[gene_str])
            #print(gene, len(gene))
            x_valid.append(np.array(gene))

        for ii in range(labels.shape[0]):
            y_valid.append(labels[ii])

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    save_dict = {
        'x': x_valid,
        'y': y_valid
    }
    save_file = valid_path.replace('.mat', '_ngram_{}.npz'.format(kernel))
    np.savez(save_file, **save_dict)

    #print(np.array(x_valid).shape)
    #print(np.array(y_valid).shape)

    #return (np.array(x_train), np.append(y_train)), (np.array(x_valid), np.array(y_valid))


if __name__ == '__main__':
    train_path ='../../data/train.mat'
    valid_path = '../../data/valid.mat'
    test_path = '../../data/test.mat'

    kernel = 8
    load_data(train_path=train_path, valid_path=valid_path, test_path=test_path)

    # (x_train, y_train), (x_valid, y_valid) =
    # print("x_train: ", x_train.shape)
    # print("y_train: ", y_train.shape)
    # print("x_valid: ", x_valid.shape)
    # print("y_valid: ", y_valid.shape)
    test_path = '../../data/test.mat'
    #loaded = h5py.File(train_path, 'r')
    loaded = sio.loadmat(test_path)
    #print(loaded)
    data = loaded['testxdata']
    labels = loaded['testdata']

    print(data.shape)
    print(labels.shape)


