
# 用于生成5gram*5的1000index序列的npz数据
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

def load_data_step_bitf_1000index_1000bp(train_path='train.mat', valid_path='valid.mat', test_path='test.mat', 
                                         dict_path = "./data/word_dict_5gram.pkl",
                                         step=5, kernel=5, shuffle=False, break_in_10w=True, **kwargs):

    import pickle
    def save_obj(obj, name):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    def load_obj(name):
        with open(name, 'rb') as f:
            return pickle.load(f)
    # 需要先加载字典
    word_dict = load_obj(dict_path)
    print("dict len : ", len(word_dict))
    print("head dict : \n", list(word_dict.items())[:5]) 
    print("step:{},kernel:{},shuffle:{},break_in_10w:{},dict_path:{}: \n".format(step,kernel,shuffle,break_in_10w,dict_path))
    
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
        for ii in tqdm(range(data.shape[2])):
            actg = np.zeros(data.shape[0])
            # 进行ACGT的转换,0，1，2，3，4
            for jj in range(data.shape[1]):
                actg = actg + data[:,jj,ii] * (1 + jj)

            actg = np.array(actg, dtype=int)

            gene = []
            # 指定0-1000，1-1000，2-1000，3-1000，4-1000，的kernel个范围
            for ee in range(kernel):
                tem_gene = []

                for jj in range(ee, len(actg),step):
                    gene_str = ''
                    # 若超过有不符合条件的跳出
                    try:
                        for kk in range(kernel):
                            gene_str += str(actg[jj + kk])
                            #print(gene_str)
                        tem_gene.append(word_dict[gene_str])
                    except Exception as e:
                        #print(gene_str)
                        # 若不足kernel长度，则对齐为kernel的长度，补充0，'422'会补为'42200'
                        if len(gene_str) < int(kernel):
                            #print("aaa")
                            gene_str = gene_str.ljust(kernel,'0')
                        #print('对齐为kernel数目，gene_str为：', gene_str)
                        tem_gene.append(word_dict[gene_str])
                        #print(e)
                        continue
                gene = gene + tem_gene
                
            x_train.append(np.array(gene))
            y_train.append(labels[ii])
            
            # 用于1w输出一次查看进度
            if index % 10000 == 0 and index > 0:
                print("Index:{}, Gene len:{}".format(index,len(gene)))

            if break_in_10w == True:
                # 用于中断该程序，20w+1，使得能够得到两次
                if index % 200001 == 0 and index > 0 and break_in_10w==True:
                    print(index,"break in 20w index")
                    break
                # 每10w保存一次
                if index % 100000 == 0 and index > 0:
                    x_train = np.array(x_train); print(np.array(x_train).shape)
                    y_train = np.array(y_train); print(np.array(y_train).shape)
                    save_dict = {
                        'x': x_train,
                        'y': y_train
                    }
                    save_file = train_path.replace('.mat', '_1000index_1000bp_{}gram_{}_newdict_all_step{}_{}.npz'.format(kernel,index,step,train_counter))
                    np.savez(save_file, **save_dict)
                    print("Saving to ", save_file)
                    x_train = []
                    y_train = []
                    train_counter += 1
            else:                
                if len(x_train) == data.shape[2]:
                    x_train = np.array(x_train); print(np.array(x_train).shape)
                    y_train = np.array(y_train); print(np.array(y_train).shape)
                    save_dict = {
                        'x': x_train,
                        'y': y_train
                    }
                    save_file = train_path.replace('.mat', '_1000index_1000bp_{}gram_{}_newdict_all_step{}_{}.npz'.format(kernel,index,step,train_counter))
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
        for ii in tqdm(range(data.shape[0])):
            actg = np.zeros(data.shape[2],dtype=int)
            for jj in range(data.shape[1]):
                actg = actg + data[ii][jj] * (1 + jj)

            actg = np.array(actg, dtype=int)

            gene = []
            # 指定0-1000，1-1000，2-1000，3-1000，4-1000，的kernel个范围
            for ee in range(kernel):
                tem_gene = []

                for jj in range(ee, len(actg),step):
                    gene_str = ''
                    # 若超过有不符合条件的跳出
                    try:
                        for kk in range(kernel):
                            gene_str += str(actg[jj + kk])
                            #print(gene_str)
                        tem_gene.append(word_dict[gene_str])
                    except Exception as e:
                        #print(gene_str)
                        # 若不足kernel长度，则对齐为kernel的长度，补充0，'422'会补为'42200'
                        if len(gene_str) < int(kernel):
                            #print("aaa")
                            gene_str = gene_str.ljust(kernel,'0')
                        #print('对齐为kernel数目，gene_str为：', gene_str)
                        tem_gene.append(word_dict[gene_str])
                        #print(e)
                        continue
                gene = gene + tem_gene
                
            x_test.append(np.array(gene))
            y_test.append(labels[ii])

            if index % 10000 == 0 and index > 0:
                print("Index:{}, Gene len:{}".format(index,len(gene)))

            if break_in_10w == True:
                if index % 200001 == 0 and index > 0 and break_in_10w==True:
                    print(index,"break in 20w index")
                    break
                if index % 100000 == 0 and index > 0:
                    x_test = np.array(x_test); print(np.array(x_test).shape)
                    y_test = np.array(y_test); print(np.array(y_test).shape)
                    save_dict = {
                        'x': x_test,
                        'y': y_test
                    }
                    save_file = test_path.replace('.mat', '_1000index_1000bp_{}gram_{}_newdict_all_step{}_{}.npz'.format(kernel,index,step,test_counter))
                    np.savez(save_file, **save_dict)
                    print("Writing to",save_file)
                    x_test = []
                    y_test = []
                    test_counter += 1
            else:
                if len(x_test) == data.shape[0]:
                    x_test = np.array(x_test); print(np.array(x_test).shape)
                    y_test = np.array(y_test); print(np.array(y_test).shape)
                    save_dict = {
                        'x': x_test,
                        'y': y_test
                    }
                    save_file = test_path.replace('.mat', '_1000index_1000bp_{}gram_{}_newdict_all_step{}_{}.npz'.format(kernel,index,step,test_counter))
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

        #选前10条测试
        #data = data[:1,:,:]
        #labels = labels[:1,:]

        index = 0
        for ii in range(data.shape[0]):
            actg = np.zeros(data.shape[2],dtype=int)
            for jj in range(data.shape[1]):
                actg = actg + data[ii][jj] * (1 + jj)

            actg = np.array(actg, dtype=int)

            gene = []
            # 指定0-1000，1-1000，2-1000，3-1000，4-1000，的kernel个范围
            for ee in range(kernel):
                tem_gene = []

                for jj in range(ee, len(actg),step):
                    gene_str = ''
                    # 若超过有不符合条件的跳出
                    try:
                        for kk in range(kernel):
                            gene_str += str(actg[jj + kk])
                            #print(gene_str)
                        tem_gene.append(word_dict[gene_str])
                    except Exception as e:
                        #print(gene_str)
                        # 若不足kernel长度，则对齐为kernel的长度，补充0，'422'会补为'42200'
                        if len(gene_str) < int(kernel):
                            #print("aaa")
                            gene_str = gene_str.ljust(kernel,'0')
                        #print('对齐为kernel数目，gene_str为：', gene_str)
                        tem_gene.append(word_dict[gene_str])
                        #print(e)
                        continue
                gene = gene + tem_gene
                #print(len(gene))

                    #print(gene_str)
                    #gene.append(word_dict[gene_str])
                #print('whole gene len :',len(gene))

            #if len(gene) != labels.shape[1]:
            #    print("请注意，长度非1000")

            x_valid.append(np.array(gene))
            y_valid.append(labels[ii])
            index+=1

            # 用于1000输出一次查看进度
            if index % 1000 == 0 and index > 0:
                #print("Index:{}, actg len:{}, actg sample: \n {}".format(index,len(actg),actg))
                #print("Index:{}, gene len:{}, gene sample: \n {}".format(index,len(gene),gene))
                print("Index:{}, gene len:{}".format(index,len(gene)))

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    save_dict = {
        'x': x_valid,
        'y': y_valid
    }
    save_file = valid_path.replace('.mat', '_1000index_1000bp_{}gram_8k_newdict_all_step{}.npz'.format(kernel,step))
    np.savez(save_file, **save_dict)
    print("Writing to",save_file)
    print(np.array(x_valid).shape)
    print(np.array(y_valid).shape)
    print("Finish valid data")




from tqdm import tqdm
import numpy as np
import scipy.io as sio
import h5py

train_path ='./data/train.mat'
valid_path = './data/valid.mat'
test_path = './data/test.mat'

dict_path = "./data/word_dict_5gram.pkl"

load_data_step_bitf_1000index_1000bp(
    train_path = train_path, valid_path = valid_path, test_path = test_path, 
    dict_path = dict_path,
    step=5, 
    kernel=5,
    shuffle=False, 
    break_in_10w=True)