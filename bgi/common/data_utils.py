
import os
import random
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def generate_batch_samples(train_data,
                           first_token_id: int,
                           last_token_id: int,
                           mask_token_id: int,
                           word_from_index: int,
                           shuffle=True,
                           batch_size: int = 32,
                           ):
    """
    生成掩码字符串
    Args:
        train_data:
        first_token_id:
        last_token_id:
        cls_token_id:
        mask_token_id:
        sep_token_id:
        word_from_index:

    Returns:

    """
    batch_len = len(train_data)
    sequence = np.zeros((batch_len, train_data.shape[1]), dtype=np.int32)
    masked_sequence = np.zeros((batch_len, train_data.shape[1]), dtype=np.int32)
    segment_id = np.zeros_like(sequence, dtype=np.int32)

    for ii in range(batch_len):
        sequence[ii] = train_data[ii]
        masked_sequence[ii] = train_data[ii]

        choice_index = np.random.choice(len(sequence[ii]), int(len(sequence[ii]) * 0.15))
        for word_pos in choice_index:
            if sequence[ii, word_pos] < word_from_index:
                continue

            dice = random.random()
            if dice < 0.8:
                masked_sequence[ii, word_pos] = mask_token_id
            elif dice < 0.9:
                masked_sequence[ii, word_pos] = random.randint(first_token_id, last_token_id)
            # else: 10% of the time we just leave the word as is
            # output_mask[ii, word_pos] = 1

    return segment_id, sequence, masked_sequence  # has_next, output_mask, sequence, segment_id, masked_sequence


def generate_batch_samples_for_classification(train_data,
                                              mask_token_id: int,
                                              probability=0.10,
                                              ):
    """
    生成掩码字符串
    Args:
        train_data:
        first_token_id:
        last_token_id:
        cls_token_id:
        mask_token_id:
        sep_token_id:
        word_from_index:

    Returns:

    """
    batch_len = len(train_data)
    masked_sequence = np.zeros((batch_len, train_data.shape[1]), dtype=np.int32)

    for ii in range(batch_len):
        masked_sequence[ii] = train_data[ii]
        choice_index = np.random.choice(len(masked_sequence[ii]), int(len(masked_sequence[ii]) * probability))
        for word_pos in choice_index:
            dice = random.random()
            if dice < 0.8:
                masked_sequence[ii, word_pos] = mask_token_id
    return masked_sequence



def load_npz_data_for_classification(file_name, ngram=3, only_one_slice=True, ngram_index=None):
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
    if str(file_name).endswith('.npz') is False or os.path.exists(file_name) is False:
        return x_data_all, y_data_all

    loaded = np.load(file_name)
    x_data = loaded['x']
    y_data = loaded['y']
    print("Load: ", file_name)
    for ii in range(ngram):
        if ngram_index is not None and ii != ngram_index:
            continue
        if only_one_slice is True:
            kk = ii
            slice_indexes = []
            max_slice_seq_len = x_data.shape[1] // ngram * ngram
            for gg in range(kk, max_slice_seq_len, ngram):
                slice_indexes.append(gg)
            x_data_slice = x_data[:, slice_indexes]
            x_data_all.append(x_data_slice)
            y_data_all.append(y_data)
        else:
            x_data_all.append(x_data)
            y_data_all.append(y_data)

    print("x_data: ", len(x_data_all))
    print("y_data: ", len(y_data_all))
    return x_data_all, y_data_all


def load_npz_data_for_pretrain(file_name, ngram=3, only_one_slice=True, ngram_index=None):
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
    if str(file_name).endswith('.npz') is False or os.path.exists(file_name) is False:
        return x_data_all, y_data_all

    loaded = np.load(file_name)
    x_data = loaded['data']
    print("load: ", file_name)
    # print("x_data: ", x_data.shape)

    for ii in range(ngram):
        if ngram_index is not None and ii != ngram_index:
            continue
        if only_one_slice is True:
            kk = ii
            slice_indexes = []
            max_slice_seq_len = x_data.shape[1] // ngram * ngram

            for gg in range(kk, max_slice_seq_len, ngram):
                slice_indexes.append(gg)

            x_data_slice = x_data[:, slice_indexes]
            x_data_all.append(x_data_slice)
        else:
            x_data_all.append(x_data)

    # print("x_data: ", len(x_data_all))
    return x_data_all


def load_npz_dataset_for_classification(record_names,
                                        batch_size=32,
                                        ngram=5,
                                        only_one_slice=True,
                                        ngram_index=None,
                                        shuffle=False,
                                        seq_len=200,
                                        num_classes=919,
                                        pool_size=24,
                                        slice_size=0,   # 默认为0则使用所有训练数据
                                        num_gpu=1,
                                        mask_token_id=2,
                                        masked=True,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                        ):
    """
    从NPZ文件中读取序列数据，并生成tf.data.DataSet
    :param record_names:
    :param batch_size:
    :param ngram:
    :param only_one_slice: 按ngram进行切片
    :param ngram_index:
    :param shuffle:
    :param seq_len:
    :param num_classes:
    :param num_parallel_calls:
    :return:
    """

    def parse_function(x, y):

        # segment_id = K.reshape(segment_id, (segment_id.shape[1], segment_id.shape[2]))
        # masked_sequence = K.reshape(masked_sequence, (masked_sequence.shape[1], sequence.shape[2]))

        print(x, y)
        masked_sequence = x
        masked_sequence = K.reshape(masked_sequence, (masked_sequence.shape[1], masked_sequence.shape[2]))  # 优化
        segment_id = K.zeros_like(masked_sequence, dtype='int64')

        y = K.cast(y, K.floatx())
        y = K.reshape(y, (y.shape[1], y.shape[2]))
        x = {
            'Input-Token': masked_sequence,
            'Input-Segment': segment_id,
        }
        y = {
            'CLS-Activation': y
        }
        return x, y

    if not isinstance(record_names, list):
        record_names = [record_names]

    total_size = 0
    # 数据生成器
    def data_generator():
        x_data_all = []
        y_data_all = []

        print("__3.1__")

        results = []
        # 使用多进程并行处理
        # pool = Pool(processes=pool_size)
        
        # LChuang: 新版本
        print("Loading data ...")
        # 先导入第一批文件，默认是50个文件一批
        record_indexes = np.arange(len(record_names))
        if shuffle == True:
            np.random.shuffle(record_indexes)

        slice_index = 0
        # 若为0则使用所有训练数据量（默认为0），否则读取按slice-size的训练数据量
        if slice_size == 0:
            slice_file_indexes = record_indexes[:len(record_names)]
        else:
            slice_file_indexes = record_indexes[:slice_size]
            
        for ii in slice_file_indexes:
            file_name = record_names[ii]
            # result = pool.apply_async(load_npz_data_for_classification,
            #                           args=(file_name,
            #                                 ngram,
            #                                 only_one_slice,
            #                                 ngram_index
            #                                 ))
            # results.append(result)

            x_data, y_data = load_npz_data_for_classification(file_name,
                                            ngram,
                                            only_one_slice,
                                            ngram_index)

            x_data_all.extend(x_data)
            y_data_all.extend(y_data)

        # pool.close()
        # pool.join()

        # HHP:旧版本
        # for file_name in record_names:
            # result = pool.apply_async(load_npz_data_for_classification,
                                      # args=(file_name,
                                            # ngram,
                                            # only_one_slice,
                                            # ngram_index
                                            # ))
            # results.append(result)
        # pool.close()
        # pool.join()

        # 汇总结果
        # for result in results:
        #     x_data, y_data = result.get()
        #     if len(x_data) > 0 and len(y_data) > 0 and len(x_data) == len(y_data):
        #         x_data_all.extend(x_data)
        #         y_data_all.extend(y_data)

        x_data_all = np.concatenate(x_data_all)
        y_data_all = np.concatenate(y_data_all)


        print("y_data_all:", type(y_data_all))
        if num_classes == 1:
            y_data_all = np.reshape(y_data_all, (y_data_all.shape[0], 1))

        print("x_data_all: ", x_data_all.shape)
        print("y_data_all: ", y_data_all.shape)

        total_size = len(x_data_all) // batch_size * batch_size
        indexes = np.arange(total_size)
        if shuffle == True:
            np.random.shuffle(indexes)

        ii = 0
        while True:
            if (ii + 1) * batch_size < total_size:  # 凑够一批次进行批量处理
                index = indexes[ii * batch_size:(ii + 1) * batch_size]
                x = x_data_all[index]
                y = y_data_all[index]

                if masked is True:
                    x = generate_batch_samples_for_classification(x,
                                              mask_token_id=mask_token_id,
                                              probability=0.10,
                                              )
                ii += 1
                yield x, y
            else:
                if shuffle is True:
                    np.random.shuffle(indexes)

                ii = 0
                index = indexes[ii * batch_size:(ii + 1) * batch_size]
                x = x_data_all[index]
                y = y_data_all[index]

                if masked is True:
                    x = generate_batch_samples_for_classification(x,
                                              mask_token_id=mask_token_id,
                                              probability=0.10,
                                              )
                ii += 1
                yield x, y

    classes_shape = tf.TensorShape([batch_size, num_classes])
    if num_classes == 1:
        classes_shape = tf.TensorShape([batch_size, 1])

    print("__1__")
    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_types=(tf.int16, tf.int16),
                                             output_shapes=(
                                                 tf.TensorShape([batch_size, seq_len]),
                                                 classes_shape
                                             ))

    print("__2__")
    # dataset = dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)

    dataset = dataset.batch(num_gpu).shuffle(1)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)

    return dataset


def load_npz_dataset_for_pretrain(record_names,
                                  batch_size=32,
                                  ngram=5,
                                  only_one_slice=True,
                                  ngram_index=None,
                                  shuffle=True,
                                  seq_len=200,
                                  num_classes=919,
                                  pool_size=48,
                                  slice_size=50,
                                  first_token_id=10,
                                  last_token_id=138,
                                  word_from_index=10,
                                  mask_token_id=2,
                                  shuffle_size=1000,
                                  num_gpu = 1,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                  ):
    """
    从NPZ文件中读取序列数据，并生成tf.data.DataSet
    :param record_names:
    :param batch_size:
    :param ngram:
    :param only_one_slice: 按ngram进行切片
    :param ngram_index:
    :param shuffle:
    :param seq_len:
    :param num_classes:
    :param num_parallel_calls:
    :return:
    """

    def parse_function(segment_id, sequence, masked_sequence):
        """
        转换为 ALBERT 的输入格式，输入inputs[Input-Token, Input-Segment], 输出Outputs[MLM-Activation]
        :param segment_id:
        :param sequence:
        :param masked_sequence:
        :return:
        """

        print("segment_id: ", segment_id)
        print("sequence: ", sequence)
        print("masked_sequence: ", masked_sequence)
        print("num_gpu: ", num_gpu)

        # 生成时，按批次生成，所以需要进行还原为[batch_size, seq_len]
        segment_id = K.reshape(segment_id, (segment_id.shape[1], segment_id.shape[2]))
        masked_sequence = K.reshape(masked_sequence, (masked_sequence.shape[1], sequence.shape[2]))

        x = {
            'Input-Token': masked_sequence,
            'Input-Segment': segment_id,
        }

        # 生成时，按批次生成，所以需要进行还原， 并转换为[batch_size, seq_len, 1]
        #sequence = K.cast(sequence, K.floatx())
        sequence = K.reshape(sequence, (sequence.shape[1], sequence.shape[2], 1))
        y = {
            'MLM-Activation': sequence
        }
        return x, y

    if not isinstance(record_names, list):
        record_names = [record_names]

    # 数据生成器
    def data_generator():
        results = []
        pool = Pool(processes=pool_size)

        print("Loading data ...")
        # 先导入第一批文件，默认是50个文件一批
        record_indexes = np.arange(len(record_names))
        if shuffle == True:
            np.random.shuffle(record_indexes)

        slice_index = 0
        slice_file_indexes = record_indexes[:slice_size]
        for ii in slice_file_indexes:
            file_name = record_names[ii]
            result = pool.apply_async(load_npz_data_for_pretrain,
                                      args=(file_name,
                                            ngram,
                                            only_one_slice,
                                            ngram_index
                                            ))
            results.append(result)
        pool.close()
        pool.join()

        # 汇总结果
        x_data_all = []
        for result in results:
            x_data = result.get()
            if len(x_data) > 0:
                x_data_all.extend(x_data)
        x_data_all = np.concatenate(x_data_all)
        slice_index += 1

        print("x_data_all: ", x_data_all.shape)
        total_size = len(x_data_all) // batch_size * batch_size
        indexes = np.arange(total_size)
        if shuffle is True:
            np.random.shuffle(indexes)

        ii = 0
        while True:
            if (ii + 1) * batch_size < total_size:
                index = indexes[ii*batch_size:(ii+1)*batch_size]
                x = x_data_all[index]

                if len(x) > 0:
                    # 生成随机掩码
                    segment_id, sequence, masked_sequence = generate_batch_samples(x,
                                                                                   first_token_id=first_token_id,
                                                                                   last_token_id=last_token_id,
                                                                                   mask_token_id=mask_token_id,
                                                                                   word_from_index=word_from_index,
                                                                                   batch_size=batch_size
                                                                                   )
                    ii += 1
                    yield segment_id, sequence, masked_sequence
                else:
                    ii += 1
            else:
                if slice_index < len(record_names) // slice_size:
                    print("-slice_index: ", slice_index)
                else:
                    # 如果文件已遍历一次，需要重新打乱文件
                    record_indexes = np.arange(len(record_names))
                    if shuffle is True:
                        np.random.shuffle(record_indexes)
                    slice_index = 0

                # 获取 下一批 数据文件
                slice_file_indexes = record_indexes[slice_size * slice_index:min(slice_size * (slice_index + 1), len(record_indexes))]
                pool = Pool(processes=pool_size)
                results = []
                print("Loading data ...")
                for ii in slice_file_indexes:
                    file_name = record_names[ii]
                    result = pool.apply_async(load_npz_data_for_pretrain,
                                              args=(file_name,
                                                    ngram,
                                                    only_one_slice,
                                                    ngram_index
                                                    ))
                    results.append(result)
                pool.close()
                pool.join()

                slice_index += 1

                # 汇总结果
                x_data_all = []
                for result in results:
                    x_data = result.get()
                    if len(x_data) > 0:
                        x_data_all.extend(x_data)
                x_data_all = np.concatenate(x_data_all)
                print("x_data_all: ", x_data_all.shape)


                total_size = len(x_data_all) // batch_size * batch_size
                indexes = np.arange(total_size)
                if shuffle is True:
                    np.random.shuffle(indexes)

                # 重置 ii 索引为 0
                ii = 0
                index = indexes[ii*batch_size:(ii+1)*batch_size]
                x = x_data_all[index]

                if len(x) > 0:
                    # 生成随机掩码
                    segment_id, sequence, masked_sequence = generate_batch_samples(x,
                                                                                   first_token_id=first_token_id,
                                                                                   last_token_id=last_token_id,
                                                                                   mask_token_id=mask_token_id,
                                                                                   word_from_index=word_from_index,
                                                                                   batch_size=batch_size
                                                                                   )
                    ii += 1
                    yield segment_id, sequence, masked_sequence
                else:
                    ii += 1


    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_types=(tf.int16, tf.int16, tf.int16),
                                             output_shapes=(
                                                 tf.TensorShape([batch_size, seq_len]),
                                                 tf.TensorShape([batch_size, seq_len]),
                                                 tf.TensorShape([batch_size, seq_len])
                                             ))
    dataset = dataset.batch(num_gpu).shuffle(1)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
    return dataset




if __name__ == '__main__':
    data_path = '/data/huadajiyin/data/hg19/gram_5_stride_1'
    # data_path = 'F:\\Research\\Data\\DeepSEA\\valid_5_gram_990'
    data_path = 'E:\\Research\\Data\\hg19\\gram_3_stride_1'
    data_path = '/home/huanghaiping/Research/Genomic/Data/hg19/gram_3_stride_1'

    files = os.listdir(data_path)
    slice_files = []
    for file_name in files:
        if str(file_name).endswith('.npz'):
            slice_files.append(os.path.join(data_path, file_name))
    dataset = load_npz_dataset_for_pretrain(slice_files, batch_size=2, ngram=3, seq_len=333, slice_size=2, ngram_index=1)

    GLOBAL_BATCH_SIZE = 64
    shuffle_size = 10000
    # train_dataset = dataset.shuffle(GLOBAL_BATCH_SIZE * shuffle_size, reshuffle_each_iteration=True).batch(GLOBAL_BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    index = 0
    for record in dataset.take(10000):
        #
        index += 1
        if index % 100 == 0:
            print(index)
            print(record)

    # tf.keras.preprocessing.image.ImageDataGenerator