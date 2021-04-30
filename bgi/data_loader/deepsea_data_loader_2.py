# -*- coding:utf-8 -*-

import pickle

import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm


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




def load_data_step_bitf_1000index_1000bp(train_path='train.mat', valid_path='valid.mat', test_path='test.mat',
                                         dict_path="../../data/word_dict_5gram.pkl", step=1, n_gram=5, shuffle=False,
                                         break_in_10w=True, **kwargs):
    """

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

    # 需要先加载字典

    # word_dict = load_obj(dict_path)
    # print("dict len : ", len(word_dict))
    # print("head dict : \n", list(word_dict.items())[:5])
    # print(word_dict)
    print("step:{},kernel:{},shuffle:{},break_in_10w:{},dict_path:{}: \n".format(step, n_gram, shuffle, break_in_10w,
                                                                                 dict_path))
    actg_value = np.array([1, 2, 3, 4])

    n_gram_value = np.ones(n_gram)
    for ii in range(n_gram):
        n_gram_value[ii] = int(n_gram_value[ii] * (10 ** (n_gram - ii - 1)))
        print(10 ** (n_gram - ii - 1))
    print("n_gram_value: ", n_gram_value)

    num_word_dict = get_word_dict_6gram()

    print("num_word_dict: ", len(num_word_dict))


    x_train = []
    y_train = []
    train_counter = 1
    if train_path is not None:

        # loaded = sio.loadmat(train_path)

        loaded = h5py.File(train_path, 'r')
        data = loaded['trainxdata']
        labels = loaded['traindata']

        print("Data: ", data.shape)  # (1000, 4, 4400000)
        print("Labels: ", labels.shape)  # (919, 4400000)

        # indexes = np.arange(data.shape[2])
        # if shuffle == True:
        #     np.random.shuffle(indexes)

        index = 0



        slice = 100000
        is_break = False
        for ii in tqdm(range(int(data.shape[2]/slice))):

            slice_data = data[:, :, ii*slice:(ii+1)*slice]
            print(slice_data.shape)

            slice_label = labels[:, ii * slice:(ii + 1) * slice]


            # 进行ACGT的转换,0，1，2，3，4
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

                        #print("10 ** (kk % n_gram): ", 10 ** (kk % n_gram))
                        actg_temp_value = actg_temp_value * (10 ** (kk % n_gram))

                    gene.append(num_word_dict.get(actg_temp_value, 0))


                x_train.append(np.array(gene))
                y_train.append(slice_label[:, jj])

                index += 1

                # 用于1w输出一次查看进度
                if index % 10000 == 0 and index > 0:
                    print("Index:{}, Gene len:{}".format(index, len(gene)))

                if break_in_10w == True:

                    # 用于中断该程序，20w+1，使得能够得到两次
                    # if index % 200001 == 0 and index > 0 and break_in_10w == True:
                    #     print(index, "break in 20w index")
                    #     is_break = True
                    #     break

                    # 每10w保存一次
                    if index % 100000 == 0 and index > 0:
                        x_train = np.array(x_train);
                        print(np.array(x_train).shape)
                        y_train = np.array(y_train);
                        print(np.array(y_train).shape)
                        save_dict = {
                            'x': x_train,
                            'y': y_train
                        }
                        save_file = train_path.replace('.mat',
                                                       '_1000index_1000bp_{}gram_{}_newdict_all_step{}_{}.npz'.format(
                                                           n_gram, index, step, train_counter))
                        np.savez(save_file, **save_dict)
                        print("Saving to ", save_file)
                        x_train = []
                        y_train = []
                        train_counter += 1
                else:
                    if len(x_train) == data.shape[2]:
                        x_train = np.array(x_train);
                        print(np.array(x_train).shape)
                        y_train = np.array(y_train);
                        print(np.array(y_train).shape)
                        save_dict = {
                            'x': x_train,
                            'y': y_train
                        }
                        save_file = train_path.replace('.mat',
                                                       '_1000index_1000bp_{}gram_{}_newdict_all_step{}_{}.npz'.format(
                                                           n_gram, index, step, train_counter))
                        np.savez(save_file, **save_dict)
                        print("Saving to ", save_file)
                        x_train = []
                        y_train = []
                        train_counter += 1


            if is_break is True:
                break

    print("Finish train data")

    x_test = []
    y_test = []
    test_counter = 1
    if test_path is not None:

        loaded = sio.loadmat(test_path)
        data = loaded['testxdata']
        labels = loaded['testdata']

        index = 0
        for ii in tqdm(range(data.shape[0])):

            actg = np.matmul(actg_value, data[ii, :, :])

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
                    actg_temp_value = actg_temp_value * (10 ** (kk % n_gram))

                gene.append(num_word_dict.get(actg_temp_value, 0))


            x_test.append(np.array(gene))
            y_test.append(labels[ii])
            index += 1


            if index % 10000 == 0 and index > 0:
                print("Index:{}, Gene len:{}".format(index, len(gene)))

            if break_in_10w == True:
                # if index % 200001 == 0 and index > 0 and break_in_10w == True:
                #     print(index, "break in 20w index")
                #     break
                if index % 100000 == 0 and index > 0:
                    x_test = np.array(x_test);
                    print(np.array(x_test).shape)
                    y_test = np.array(y_test);
                    print(np.array(y_test).shape)
                    save_dict = {
                        'x': x_test,
                        'y': y_test
                    }
                    save_file = test_path.replace('.mat',
                                                  '_1000index_1000bp_{}gram_{}_newdict_all_step{}_{}.npz'.format(n_gram,
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
                    x_test = np.array(x_test);
                    print(np.array(x_test).shape)
                    y_test = np.array(y_test);
                    print(np.array(y_test).shape)
                    save_dict = {
                        'x': x_test,
                        'y': y_test
                    }
                    save_file = test_path.replace('.mat',
                                                  '_1000index_1000bp_{}gram_{}_newdict_all_step{}_{}.npz'.format(n_gram,
                                                                                                                 index,
                                                                                                                 step,
                                                                                                                 test_counter))
                    np.savez(save_file, **save_dict)
                    print("Writing to", save_file)
                    x_test = []
                    y_test = []
                    test_counter += 1

    print("Finish test data")

    x_valid = []
    y_valid = []
    if valid_path is not None:
        loaded = sio.loadmat(valid_path)
        data = loaded['validxdata']
        labels = loaded['validdata']

        # 选前10条测试
        # data = data[:1,:,:]
        # labels = labels[:1,:]

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

            # if len(gene) != labels.shape[1]:
            #    print("请注意，长度非1000")

            x_valid.append(np.array(gene))
            y_valid.append(labels[ii])
            index += 1

            # 用于1000输出一次查看进度
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
        save_file = valid_path.replace('.mat', '_1000index_1000bp_{}gram_8k_newdict_all_step{}.npz'.format(n_gram, step))
        np.savez(save_file, **save_dict)
        print("Writing to", save_file)
        print(np.array(x_valid).shape)
        print(np.array(y_valid).shape)
        print("Finish valid data")


if __name__ == '__main__':
    prefix = "/data/HPhuang_data/DeepSEA/"
    # prefix = "F:\\Research\\Data\\DeepSEA\\deepsea_train\\"
    train_path = prefix + 'train.mat'
    valid_path = prefix + 'valid.mat'
    test_path = prefix + 'test.mat'

    dict_path = "../../data/word_dict_5gram.pkl"

    load_data_step_bitf_1000index_1000bp(
        train_path=train_path, valid_path=valid_path, test_path=test_path,
        dict_path=dict_path,
        step=1,
        n_gram=1,
        shuffle=False,
        break_in_10w=False)

    # word_dict = get_word_dict_5gram()
    # print(word_dict)
