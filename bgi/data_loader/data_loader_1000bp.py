

# 生成10W的顺序1000bp序列（非onehot），用01234分别表示NAGCT碱基

import pickle

import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm

def load_data_step_bitf_01234_1000bp(train_path='train.mat', valid_path='valid.mat', test_path='test.mat', 
                   shuffle=False, break_in_10w=True, **kwargs):

    import pickle
    def save_obj(obj, name):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    def load_obj(name):
        with open(name, 'rb') as f:
            return pickle.load(f)
    # 需要先加载字典
    #word_dict = load_obj(dict_path)
    #print("dict len : ", len(word_dict))
    #print("head dict : \n", list(word_dict.items())[:5]) 
    #print("step:{},kernel:{},shuffle:{},break_in_10w:{},dict_path:{}: \n".format(step,kernel,shuffle,break_in_10w,dict_path))
    
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
            # 进行ACGT的转换,0，1，2，3，4
            for jj in range(data.shape[1]):
                actg = actg + data[:,jj,ii] * (1 + jj)
            actg = np.array(actg, dtype=int)
            
            #print("actg: ", actg)
            x_train.append(np.array(actg))
            y_train.append(labels[:, ii])
            
            # 用于1w输出一次查看进度
            if index % 10000 == 0 and index > 0:
                print("Index:{}, actg len:{}".format(index,len(actg)))

            if break_in_10w == True:
                # 用于中断该程序，20w+1，使得能够得到两次
                if index % 100001 == 0 and index > 0 and break_in_10w==True:
                    print(index,"break in 20w index")
                    break
                # 每10w保存一次
                if index % 100000 == 0 and index > 0:
                    x_train = np.array(x_train)
                    y_train = np.array(y_train)
                    save_dict = {
                        'x': x_train,
                        'y': y_train
                    }
                    save_file = train_path.replace('.mat', '_01234_1000bp_{}_newdict_all_{}.npz'.format(index,train_counter))
                    np.savez(save_file, **save_dict)
                    print("Saving to ", save_file)
                    x_train = []
                    y_train = []
                    train_counter += 1
            else:                
                if len(x_train) == data.shape[2]:
                    x_train = np.array(x_train)
                    y_train = np.array(y_train)
                    save_dict = {
                        'x': x_train,
                        'y': y_train
                    }
                    save_file = train_path.replace('.mat', '_01234_1000bp_{}_newdict_all_{}.npz'.format(index,train_counter))
                    np.savez(save_file, **save_dict)
                    print("Saving to ", save_file)
                    x_train = []
                    y_train = []
                    train_counter += 1
            index += 1
    print("Finish train data")

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
                actg = actg + data[ii][jj] * (1 + jj)
            actg = np.array(actg, dtype=int)

            # print("actg: ", actg)
            x_test.append(np.array(actg))
            y_test.append(labels[ii])

            if index % 10000 == 0 and index > 0:
                print("Index:{}, actg len:{}".format(index,len(actg)))

            if break_in_10w == True:
                if index % 100001 == 0 and index > 0 and break_in_10w==True:
                    print(index,"break in 20w index")
                    break
                if index % 100000 == 0 and index > 0:
                    x_test = np.array(x_test)
                    y_test = np.array(y_test)
                    save_dict = {
                        'x': x_test,
                        'y': y_test
                    }
                    save_file = test_path.replace('.mat', '_01234_1000bp_{}_newdict_all_{}.npz'.format(index,test_counter))
                    np.savez(save_file, **save_dict)
                    print("Writing to",save_file)
                    x_test = []
                    y_test = []
                    test_counter += 1
            else:
                if len(x_test) == data.shape[0]:
                    x_test = np.array(x_test)
                    y_test = np.array(y_test)
                    save_dict = {
                        'x': x_test,
                        'y': y_test
                    }
                    save_file = test_path.replace('.mat', '_01234_1000bp_{}_newdict_all_{}.npz'.format(index,test_counter))
                    np.savez(save_file, **save_dict)
                    print("Writing to",save_file)
                    x_test = []
                    y_test = []
                    test_counter += 1
            index += 1
    print("Finish test data")


    x_valid = []
    y_valid = []
    if valid_path is not None:
        loaded = sio.loadmat(valid_path)
        data = loaded['validxdata']
        labels = loaded['validdata']

        for ii in range(data.shape[0]):
            actg = np.zeros(data.shape[2])
            for jj in range(data.shape[1]):
                actg = actg + data[ii][jj] * (1 + jj)
            actg = np.array(actg, dtype=int)

            x_valid.append(np.array(actg))

        for ii in range(labels.shape[0]):
            y_valid.append(labels[ii])

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    save_dict = {
        'x': x_valid,
        'y': y_valid
    }
    save_file = valid_path.replace('.mat', '_01234_1000bp_8k_newdict_all.npz')
    np.savez(save_file, **save_dict)
    print("Writing to",save_file)
    print(np.array(x_valid).shape)
    print(np.array(y_valid).shape)
    print("Finish valid data")

if __name__ == '__main__':
    
    train_path='./data/train.mat'
    valid_path='./data/valid.mat'
    test_path='./data/test.mat'
    
    # 生成10W的顺序1000bp序列（非onehot），用01234分别表示NAGCT碱基
    load_data_step_bitf_01234_1000bp(train_path='./data/train.mat', 
                                     valid_path='./data/valid.mat',
                                     test_path='./data/test.mat', shuffle=False, break_in_10w=True)
                                     
                                     
                                     