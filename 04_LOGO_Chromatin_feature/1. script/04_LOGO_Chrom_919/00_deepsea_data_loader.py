# -*- coding:utf-8 -*-

import argparse
import os
import sys
from multiprocessing import Pool
import time

import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import threading

sys.path.append("../../")
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number


def preccess_data(slice,
                  slice_index,
                  actg_value,
                  step,
                  n_gram,
                  n_gram_value,
                  num_word_dict,
                  train_counter,
                  train_path,
                  output_path,
                  slice_data=None,
                  slice_label=None):
    """
    Process data in batches and generate npz files
    :param slice:
    :param slice_index:
    :param actg_value:
    :param step:
    :param n_gram:
    :param n_gram_value:
    :param num_word_dict:
    :param train_counter:
    :param train_path:
    :return:
    """

    #
    x_train = []
    y_train = []

    # time.sleep( 10 )

    print("slice_index: ", slice_index)
    print(slice_data.shape)

    # slice_label = labels[:, slice_index * slice:(slice_index + 1) * slice]

    # AGCT is converted to 1, 2, 3, 4
    for jj in range(slice):
        actg = np.matmul(slice_data[:, :, jj], actg_value)
        # for ss in range(n_gram):
        gene = []
        for kk in range(0, len(actg), step):
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

        x_train.append(np.array(gene))
        y_train.append(slice_label[:, jj])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(np.array(y_train).shape)
    print(np.array(x_train).shape)
    save_dict = {
        'x': x_train,
        'y': y_train
    }

    train_output_path = os.path.join(output_path, 'train_{}_gram'.format(n_gram))
    if os.path.exists(train_output_path) is False:
        os.makedirs(train_output_path)

    bp = x_train.shape[1]
    save_file = os.path.join(train_output_path, 'deepsea_train_{}_bp_{}_gram_{}_step_{}.npz'.format(
        bp, n_gram, step, train_counter))
    np.savez_compressed(save_file, **save_dict)
    print("Saving to ", save_file)
    del x_train
    del y_train
    return "Finish"


def load_data_from_mat_file(train_path='train.mat',
                            valid_path='valid.mat',
                            test_path='test.mat',
                            step=1,
                            n_gram=5,
                            shuffle=False,
                            break_in_10w=True,
                            slice=200000,
                            pool_size=8,
                            output_path='',
                            **kwargs):
    """
    Import Data
    :param train_path:
    :param valid_path:
    :param test_path:
    :param dict_path:
    :param step:
    :param kernel:
    :param shuffle:
    :param break_in_10w:
    :param kwargs:
    :return:
    """

    print("step:{}, kernel:{}, shuffle:{}, break_in_10w:{} \n".format(step, n_gram, shuffle, break_in_10w))

    actg_value = np.array([1, 2, 3, 4])
    n_gram_value = np.ones(n_gram)
    for ii in range(n_gram):
        n_gram_value[ii] = int(n_gram_value[ii] * (10 ** (n_gram - ii - 1)))
    print("n_gram_value: ", n_gram_value)

    num_word_dict = get_word_dict_for_n_gram_number(n_gram=n_gram)

    # Train set
    if train_path is not None:
        loaded = h5py.File(train_path, 'r')
        data = loaded['trainxdata']
        labels = loaded['traindata']

        print("Data: ", data.shape)  # (1000, 4, 4400000)
        print("Labels: ", labels.shape)  # (919, 4400000)

        pool = Pool(processes=pool_size)
        for ii in range(int(data.shape[2] / slice)):
            print("__1__")
            print(ii * slice, (ii + 1) * slice)

            print("__2__")

            slice_data = data[:, :, ii * slice:min((ii + 1) * slice, data.shape[2])]
            slice_label = labels[:, ii * slice:min((ii + 1) * slice, data.shape[2])]

            pool.apply_async(preccess_data, args=(
            slice, ii, actg_value, step, n_gram, n_gram_value, num_word_dict, ii, train_path, output_path, slice_data,
            slice_label))

        pool.close()
        pool.join()
    print("Finish train data")

    x_test = []
    y_test = []
    test_counter = 1

    # Test set
    if test_path is not None:

        loaded = sio.loadmat(test_path)
        data = loaded['testxdata']
        labels = loaded['testdata']

        test_output_path = os.path.join(output_path, 'test_{}_gram'.format(n_gram))
        if os.path.exists(test_output_path) is False:
            os.makedirs(test_output_path)

        index = 0
        for ii in tqdm(range(data.shape[0])):
            actg = np.matmul(actg_value, data[ii, :, :])
            gene = []
            for kk in range(0, len(actg), step):
                actg_temp_value = 0
                if kk + n_gram <= len(actg):
                    actg_temp_value = np.dot(actg[kk:kk + n_gram], n_gram_value)
                    actg_temp_value = int(actg_temp_value)
                else:
                    for gg in range(kk, len(actg)):
                        actg_temp_value += actg[gg] * (10 ** (n_gram - gg % n_gram - 1))
                    actg_temp_value = actg_temp_value * (10 ** (kk % n_gram))
                gene.append(num_word_dict.get(actg_temp_value, 0))

            x_test.append(np.array(gene))
            y_test.append(labels[ii])
            index += 1

            if index % 10000 == 0 and index > 0:
                print("Index:{}, Gene len:{}".format(index, len(gene)))

            if break_in_10w == True:
                if index % 100000 == 0 and index > 0:
                    x_test = np.array(x_test);
                    print(np.array(x_test).shape)
                    y_test = np.array(y_test);
                    print(np.array(y_test).shape)
                    save_dict = {
                        'x': x_test,
                        'y': y_test
                    }
                    save_file = os.path.join(test_output_path,
                                             'deepsea_test_{}_bp_{}_gram_{}_{}_step_{}.npz'.format((x_test.shape[1]),
                                                                                                   n_gram,
                                                                                                   index,
                                                                                                   step,
                                                                                                   test_counter))
                    np.savez(save_file, **save_dict)
                    print("Writing to", save_file)
                    x_test = []
                    y_test = []
                    test_counter += 1
            else:
                if len(x_test) == data.shape[0]:
                    x_test = np.array(x_test)
                    y_test = np.array(y_test)

                    print(np.array(x_test).shape)
                    print(np.array(y_test).shape)

                    save_dict = {
                        'x': x_test,
                        'y': y_test
                    }
                    save_file = os.path.join(test_output_path,
                                             'deepsea_test_{}_bp_{}_gram_{}_step_{}.npz'.format((x_test.shape[1]),
                                                                                                n_gram,
                                                                                                index,
                                                                                                step,
                                                                                                test_counter))
                    np.savez_compressed(save_file, **save_dict)
                    print("Writing to", save_file)
                    del x_test
                    del y_test
                    test_counter += 1
        print("Finish test data")

    x_valid = []
    y_valid = []

    # Valid Set
    if valid_path is not None:
        loaded = sio.loadmat(valid_path)
        data = loaded['validxdata']
        labels = loaded['validdata']

        index = 0
        for ii in range(data.shape[0]):
            actg = np.matmul(actg_value, data[ii, :, :])
            gene = []
            for kk in range(0, len(actg), step):
                actg_temp_value = 0
                if kk + n_gram <= len(actg):
                    actg_temp_value = np.dot(actg[kk:kk + n_gram], n_gram_value)
                    actg_temp_value = int(actg_temp_value)
                else:
                    for gg in range(kk, len(actg)):
                        actg_temp_value += actg[gg] * (10 ** (n_gram - gg % n_gram - 1))
                    actg_temp_value = actg_temp_value * (10 ** (kk % n_gram))
                gene.append(num_word_dict.get(actg_temp_value, 0))

            x_valid.append(np.array(gene))
            y_valid.append(labels[ii])
            index += 1

            # Used for 1000 output to view the progress at a time
            if index % 1000 == 0 and index > 0:
                # print("Index:{}, actg len:{}, actg sample: \n {}".format(index,len(actg),actg))
                # print("Index:{}, gene len:{}, gene sample: \n {}".format(index,len(gene),gene))
                print("Index:{}, gene len:{}".format(index, len(gene)))

        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)
        save_dict = {
            'x': x_valid,
            'y': y_valid
        }

        valid_output_path = os.path.join(output_path, 'valid_{}_gram'.format(n_gram))
        if os.path.exists(valid_output_path) is False:
            os.makedirs(valid_output_path)

        save_file = os.path.join(valid_output_path,
                                 'deepsea_valid_{}_bp_{}_gram_8k_{}_step.npz'.format((x_valid.shape[1]), n_gram, step))
        np.savez_compressed(save_file, **save_dict)
        print("Writing to", save_file)
        print(np.array(x_valid).shape)
        print(np.array(y_valid).shape)
        print("Finish valid data")


if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='A data preprocessing of the Transformer language model in Genomics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--data', type=str, required=True, metavar='PATH',
        help='A path of DeepSEA file.')
    _argparser.add_argument(
        '--output', type=str, default='transformer_gene', metavar='NAME',
        help='A path which save processed file')
    _argparser.add_argument(
        '--ngram', type=int, default=3, metavar='INTEGER',
        help='ngram')
    _argparser.add_argument(
        '--stride', type=int, default=1, metavar='INTEGER',
        help='Stride of ngram')
    _argparser.add_argument(
        '--slice', type=int, default=100000, metavar='INTEGER',
        help='Slice of train set')
    _argparser.add_argument(
        '--pool-size', type=int, default=4, metavar='INTEGER',
        help='Pool size of multi-thread')

    _args = _argparser.parse_args()
    prefix = _args.data  # file name
    output_path = _args.output  #
    ngram = _args.ngram
    stride = _args.stride
    slice = _args.slice
    pool_size = _args.pool_size

    train_path = os.path.join(prefix, 'train.mat')
    valid_path = os.path.join(prefix, 'valid.mat')
    test_path = os.path.join(prefix, 'test.mat')

    load_data_from_mat_file(
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        step=stride,
        n_gram=ngram,
        shuffle=False,
        break_in_10w=False,
        output_path=output_path,
        slice=slice,
        pool_size=pool_size,
    )
