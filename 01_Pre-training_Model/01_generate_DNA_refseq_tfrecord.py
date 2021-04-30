import os
import random
from multiprocessing import Pool
import argparse
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

sys.path.append("../")
from bgi.common.refseq_utils import get_word_dict_for_n_gram_alphabet


def generate_samples(train_data,
                     first_token_id: int,
                     last_token_id: int,
                     cls_token_id: int,
                     mask_token_id: int,
                     sep_token_id: int,
                     word_from_index: int,
                     is_sep_mask: bool = False,
                     shuffle=True,
                     batch_size: int = 32,
                     output_path: str = '',
                     file_name: str = ''
                     ):
    """
    Generate masked sequence
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

    if is_sep_mask:
        sequence = np.zeros((batch_len, train_data.shape[1] + 2), dtype=np.int32)
        masked_sequence = np.zeros((batch_len, train_data.shape[1] + 2), dtype=np.int32)

        indexes = np.arange(len(train_data))
        if shuffle is True:
            np.random.shuffle(indexes)

        for ii in range(batch_len):
            index = indexes[ii]
            sequence[ii, 0] = cls_token_id
            sequence[ii, 1:-1] = train_data[index, :]
            sequence[ii, -1] = sep_token_id
            masked_sequence[ii] = sequence[ii]

            # Mask 15% tokens
            for word_pos in range(1, len(sequence[ii]) - 1):
                if random.random() < 0.15:
                    dice = random.random()
                    if dice < 0.8:
                        masked_sequence[ii, word_pos + 1] = mask_token_id
                    elif dice < 0.9:
                        masked_sequence[ii, word_pos + 1] = random.randint(first_token_id, last_token_id)

    else:
        sequence = np.zeros((batch_len, train_data.shape[1]), dtype=np.int32)
        masked_sequence = np.zeros((batch_len, train_data.shape[1]), dtype=np.int32)

        indexes = np.arange(len(train_data))
        if shuffle is True:
            np.random.shuffle(indexes)

        for ii in range(batch_len):
            index = indexes[ii]
            sequence[ii] = train_data[index]
            masked_sequence[ii] = train_data[index]

            choice_index = np.random.choice(len(sequence[ii]), int(len(sequence[ii]) * 0.15))
            for word_pos in choice_index:
                if sequence[ii, word_pos] < word_from_index:
                    continue

                dice = random.random()
                if dice < 0.8:
                    masked_sequence[ii, word_pos] = mask_token_id
                elif dice < 0.9:
                    masked_sequence[ii, word_pos] = random.randint(first_token_id, last_token_id)

    print("Masked_sequence: ", masked_sequence.shape)
    print("Sequence: ", sequence.shape)

    # Save to file
    serialized_instances = tfrecord_serialize([masked_sequence, sequence], ['masked_sequence', 'sequence'])
    write_to_tfrecord(os.path.join(output_path, file_name), serialized_instances)
    print("Save: ", os.path.join(output_path, file_name))


def tfrecord_serialize(instances, instance_keys):
    """
    Converted to tfrecord format record, waiting to be written to the file
    """

    def create_feature(x):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

    serialized_instances = []
    for instance in zip(*instances):
        if len(instance) != len(instance_keys):
            continue

        features = {
            k: create_feature(v)
            for k, v in zip(instance_keys, instance)
        }
        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        serialized_instance = tf_example.SerializeToString()
        serialized_instances.append(serialized_instance)

    return serialized_instances


def write_to_tfrecord(record_name, serialized_instances):
    options = tf.io.TFRecordOptions(
        compression_type=tf.io.TFRecordOptions.compression_type_map.get('ZLIB'))

    writer = tf.io.TFRecordWriter(record_name, options)
    for serialized_instance in serialized_instances:
        writer.write(serialized_instance)


def load_tfrecord(record_names, sequence_length=100, batch_size=32):
    """
    Load tfrecord and parse_function
    """

    def parse_function(serialized):
        features = {
            'masked_sequence': tf.io.FixedLenFeature([sequence_length], tf.int64),
            'sequence': tf.io.FixedLenFeature([sequence_length], tf.int64),
        }
        features = tf.io.parse_single_example(serialized, features)
        masked_sequence = features['masked_sequence']
        segment_id = K.zeros_like(masked_sequence, dtype='int64')
        sequence = features['sequence']
        x = {
            'Input-Token': masked_sequence,
            'Input-Segment': segment_id,
        }
        y = sequence
        return x, y

    if not isinstance(record_names, list):
        record_names = [record_names]

    dataset = tf.data.TFRecordDataset(record_names)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def prepare_pretrain_data(train_data_path: str,
                          output_path: str,
                          pool_size: int = 8,
                          ngram: int = 3,
                          stride: int = 1,
                          first_token_id: int = 10,
                          last_token_id: int = 10,
                          CLS_ID: int = 1,
                          MASK_ID: int = 2,
                          PAD_ID: int = 3,
                          word_from_index: int = 10,
                          is_sep_mask: bool = False,
                          shuffle=True,
                          batch_size: int = 32,
                          only_one_slice: bool = True,
                          ):
    files = os.listdir(train_data_path)

    # Generate data in parallel
    pool = Pool(processes=pool_size)
    results = []
    for file_name in files:

        if str(file_name).endswith('.npz') is False:
            continue
        train_file = os.path.join(train_data_path, file_name)
        print("File: ", train_file)
        loaded = np.load(train_file)
        x_train = loaded['data']

        tf_file_name = file_name.replace('.npz', '.tfrecord')
        # Read all data
        if only_one_slice is True:
            # kk = random.randint(0, int(ngram) - 1)
            for kk in range(ngram):
                slice_indexes = []
                max_slice_seq_len = x_train.shape[1] // ngram * ngram
                for gg in range(kk, max_slice_seq_len, ngram):
                    slice_indexes.append(gg)

                print("max_slice_seq_len: ", max_slice_seq_len, kk)
                print("slice_indexes: ", len(slice_indexes))
                suffix = '_{}.tfrecord'.format(str(kk))
                tf_file_name = tf_file_name.replace('.tfrecord', suffix)

                pool.apply_async(generate_samples,
                                 args=(x_train[:, slice_indexes],
                                       first_token_id,
                                       last_token_id,
                                       CLS_ID,
                                       MASK_ID,
                                       PAD_ID,
                                       word_from_index,
                                       is_sep_mask,
                                       shuffle,
                                       batch_size,
                                       output_path,
                                       tf_file_name,
                                       ))
                # results.append(result)

        else:
            pool.apply_async(generate_samples,
                             args=(x_train,
                                   first_token_id,
                                   last_token_id,
                                   CLS_ID,
                                   MASK_ID,
                                   PAD_ID,
                                   word_from_index,
                                   is_sep_mask,
                                   shuffle,
                                   batch_size,
                                   output_path,
                                   tf_file_name,
                                   ))
            # results.append(result)
    pool.close()
    pool.join()


if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='A data preprocessing of the Transformer language model in Genomics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--data', type=str, required=True, metavar='PATH',
        help='A path of hg19/38 file.')
    _argparser.add_argument(
        '--output', type=str, default='transformer_gene', metavar='NAME',
        help='A path which save processed file')
    _argparser.add_argument(
        '--chunk-size', type=int, default=10000, metavar='INTEGER',
        help='chunk size')
    _argparser.add_argument(
        '--seq-size', type=int, default=1000, metavar='INTEGER',
        help='Sequence size')
    _argparser.add_argument(
        '--seq-stride', type=int, default=100, metavar='INTEGER',
        help='Sequence stride size')
    _argparser.add_argument(
        '--ngram', type=int, default=3, metavar='INTEGER',
        help='NGram')
    _argparser.add_argument(
        '--stride', type=int, default=1, metavar='INTEGER',
        help='Stride')
    _argparser.add_argument(
        '--slice-size', type=int, default=100000, metavar='INTEGER',
        help='Slice size')
    _argparser.add_argument(
        '--hg-name', type=str, default='hg19', metavar='NAME',
        help='hg name')
    _argparser.add_argument(
        '--pool-size', type=int, default=4, metavar='INTEGER',
        help='Pool size')

    _args = _argparser.parse_args()

    data_file = _args.data
    output_path = _args.output
    # chunk_size = _args.chunk_size
    # seq_size = _args.seq_size
    # seq_stride = _args.seq_stride
    ngram = _args.ngram
    stride = _args.stride
    # slice_size = _args.slice_size
    # hg_name = _args.hg_name
    pool_size = _args.pool_size

    word_index_from = 10
    word_dict_alphabet = get_word_dict_for_n_gram_alphabet(n_gram=ngram, word_index_from=10)
    print("word_dict_alphabet: ", len(word_dict_alphabet))

    last_token_id = len(word_dict_alphabet) + word_index_from

    batch_size = 32
    num_gpu = 8
    global_batch_size = 32 * num_gpu

    prepare_pretrain_data(train_data_path=data_file,
                          output_path=output_path,
                          pool_size=pool_size,
                          ngram=ngram,
                          stride=stride,
                          first_token_id=word_index_from,
                          last_token_id=last_token_id,
                          word_from_index=word_index_from,
                          batch_size=global_batch_size
                          )
