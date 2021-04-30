import tensorflow as tf
import numpy as np
import h5py
import os
import gzip
import random


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def serialize_example(x_feature, x_weight, y_label):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'feature': _int64_feature(x_feature),
        'weight': _float_feature(x_weight),
        'label': _int64_feature(y_label),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def serialize_seq_example(x_feature, x_weight, seq, alt_seq, alt_type, y_label):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'feature': _int64_feature(x_feature),
        'weight': _float_feature(x_weight),
        'label': _int64_feature(y_label),
        'seq': _int64_feature(seq),
        'alt_seq': _int64_feature(alt_seq),
        'alt_type': _int64_feature(alt_type),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_tfrecord(data_file, tfrecord_file):
    """
    Create TF Record
    :param data_file:
    :param tfrecord_file:
    :return:
    """
    if os.path.exists(data_file) is False or str(data_file).endswith('.gz') is False:
        return

    train_file = tfrecord_file + '_train.tfrecord'
    valid_file = tfrecord_file + '_valid.tfrecord'
    test_file = tfrecord_file + '_test.tfrecord'

    train_writer = tf.io.TFRecordWriter(train_file)
    valid_writer = tf.io.TFRecordWriter(valid_file)
    test_writer = tf.io.TFRecordWriter(test_file)

    counter = 0

    with gzip.open(data_file, 'r') as pf:
        for line in pf:
            # Remove b'and \n'
            line = str(line)
            line = line[2:-3]

            tokens = line.split(',')
            if len(tokens) < 2:
                continue

            y = tokens[0]
            x = tokens[1:]
            x = np.array(x, dtype=np.float)

            # Filter out features equal to 0
            indexes = np.where(x > 0.0)
            values = x[indexes]

            # Feature Index
            features = np.array(indexes)
            features = np.reshape(features, (features.shape[1]))

            # Eigenvalues
            values = np.array(values, dtype=np.float)

            # Tags
            y = np.array([int(y)])

            example = serialize_example(features, values, y)
            dice = random.random()
            if dice < 0.9:
                train_writer.write(example)
            elif dice < 0.95:
                valid_writer.write(example)
            else:
                test_writer.write(example)

            counter += 1
            if counter % 10000 == 0:
                print(counter)

            # if counter > 200000:
            #     break


def create_tfrecord_with_sequence(data_file, seq_file, tfrecord_file):
    """
    Create TF Record
    :param data_file:
    :param tfrecord_file:
    :return:
    """

    if os.path.exists(data_file) is False or str(data_file).endswith('.gz') is False:
        return

    if os.path.exists(seq_file) is False or str(seq_file).endswith('.npz') is False:
        return

    loaded = np.load(seq_file, allow_pickle=True)
    seq_data = loaded['x']
    alt_seq_data = loaded['alt']
    alt_type_data = loaded['type']

    counter = 0
    row_index = 0

    train_file = tfrecord_file + '_train.tfrecord'
    valid_file = tfrecord_file + '_valid.tfrecord'
    test_file = tfrecord_file + '_test.tfrecord'

    train_writer = tf.io.TFRecordWriter(train_file)
    valid_writer = tf.io.TFRecordWriter(valid_file)
    test_writer = tf.io.TFRecordWriter(test_file)

    with gzip.open(data_file, 'r') as pf:
        for line in pf:
            # Remove b'and \n'
            line = str(line)
            line = line[2:-3]

            tokens = line.split(',')
            if len(tokens) < 2:
                continue

            y = tokens[0]
            x = tokens[1:]
            x = np.array(x, dtype=np.float)

            # Filter out features equal to 0
            indexes = np.where(x != 0.0)
            values = x[indexes]

            # Feature Index
            features = np.array(indexes)
            features = np.reshape(features, (features.shape[1]))

            # Eigenvalues
            values = np.array(values, dtype=np.float)

            # Tags
            y = np.array([int(y)])

            seq = seq_data[row_index]
            alt = alt_seq_data[row_index]
            type = alt_type_data[row_index]

            example = serialize_seq_example(features, values, seq, alt, type, y)

            dice = random.random()
            if dice < 0.9:
                train_writer.write(example)
            elif dice < 0.95:
                valid_writer.write(example)
            else:
                test_writer.write(example)

            counter += 1
            if counter % 100000 == 0:
                print(counter)

            row_index += 1

            # if counter > 200000:
            #     break


def create_tfrecord_only_sequence(seq_file, tfrecord_file, ngram=3, only_one_file=True):
    """
    Create TF Record
    :param data_file:
    :param tfrecord_file:
    :return:
    """

    # if os.path.exists(data_file) is False or str(data_file).endswith('.gz') is False:
    #     return

    if os.path.exists(seq_file) is False or str(seq_file).endswith('.npz') is False:
        return

    loaded = np.load(seq_file, allow_pickle=True)
    seq_data = loaded['x']
    alt_seq_data = loaded['alt']
    alt_type_data = loaded['type']
    label_data = loaded['label']

    counter = 0
    row_index = 0

    if only_one_file:
        data_file = seq_file.replace('.npz', '.tfrecord')
        data_writer = tf.io.TFRecordWriter(data_file)
    else:
        # train_file = tfrecord_file + '_train.tfrecord'
        valid_file = tfrecord_file + '_valid_{}_gram.tfrecord'.format(str(ngram))
        test_file = tfrecord_file + '_test_{}_gram.tfrecord'.format(str(ngram))

        # train_writer = tf.io.TFRecordWriter(train_file)
        valid_writer = tf.io.TFRecordWriter(valid_file)
        test_writer = tf.io.TFRecordWriter(test_file)


    for seq, alt, type, y in zip(seq_data, alt_seq_data, alt_type_data, label_data):
        # Feature Index
        features = np.array([0])
        # Eigenvalues
        values = np.array([0], dtype=np.float)

        # Tags
        y = np.array([int(y)])

        seq = seq_data[row_index]
        alt = alt_seq_data[row_index]
        type = alt_type_data[row_index]

        example = serialize_seq_example(features, values, seq, alt, type, y)
        if only_one_file:
            data_writer.write(example)
        else:
            dice = random.random()
            if dice < 0.5:
                valid_writer.write(example)
            else:
                test_writer.write(example)

        counter += 1
        if counter % 100000 == 0:
            print(counter)

        row_index += 1


def create_tfrecord_only_sequence_all(seq_file, tfrecord_file, ngram=3):
    """
    Create TF Record
    :param data_file:
    :param tfrecord_file:
    :return:
    """

    # if os.path.exists(data_file) is False or str(data_file).endswith('.gz') is False:
    #     return

    if os.path.exists(seq_file) is False or str(seq_file).endswith('.npz') is False:
        return

    loaded = np.load(seq_file, allow_pickle=True)
    seq_data = loaded['x']
    alt_seq_data = loaded['alt']
    alt_type_data = loaded['type']
    label_data = loaded['label']

    counter = 0
    row_index = 0

    train_file = tfrecord_file + '_train_{}_gram.tfrecord'.format(str(ngram))
    # valid_file = tfrecord_file + '_valid_{}_gram.tfrecord'.format(str(ngram))
    # test_file = tfrecord_file + '_test_{}_gram.tfrecord'.format(str(ngram))

    train_writer = tf.io.TFRecordWriter(train_file)
    # valid_writer = tf.io.TFRecordWriter(valid_file)
    # test_writer = tf.io.TFRecordWriter(test_file)


    for seq, alt, type, y in zip(seq_data, alt_seq_data, alt_type_data, label_data):
        features = np.array([0])
        values = np.array([0], dtype=np.float)
        y = np.array([int(y)])

        seq = seq_data[row_index]
        alt = alt_seq_data[row_index]
        type = alt_type_data[row_index]

        example = serialize_seq_example(features, values, seq, alt, type, y)

        train_writer.write(example)
        dice = random.random()
        # if dice < 0.5:
        #     valid_writer.write(example)
        # else:
        #     test_writer.write(example)

        counter += 1
        if counter % 100000 == 0:
            print(counter)

        row_index += 1

if __name__ == '__main__':

    # train_data_file = 'D:\\Research\\Data\\Genomics\\CADD\\noncoding\\simulation_SNV_noncoding.csv.gz'
    # nfolds = 5
    # for fold in range(nfolds):
    #     tfrecord_file = './data/clinvar_InDel_seq'
    #     seq_file = './data/fold{}/clinvar_InDel_fold{}_test_gram_3_stride_1_slice_2935.npz'.format(str(fold), str(fold))
    #     ngram = 3
    #     create_tfrecord_only_sequence(seq_file, tfrecord_file, ngram=ngram)
    #
    #     seq_file = './data/fold{}/clinvar_InDel_fold{}_valid_gram_3_stride_1_slice_2934.npz'.format(str(fold), str(fold))
    #     ngram = 3
    #     create_tfrecord_only_sequence(seq_file, tfrecord_file, ngram=ngram)
    #

    tfrecord_file = '/data/CADD/GRCh37/humanDerived_InDels_seq'
    seq_file = '/data/CADD/GRCh37/humanDerived_InDels_gram_3_stride_1_slice_1837498.npz'
    ngram = 3
    create_tfrecord_only_sequence_all(seq_file, tfrecord_file, ngram=ngram)

    tfrecord_file = '/data/CADD/GRCh37/simulation_InDels_seq'
    seq_file = '/data/CADD/GRCh37/simulation_InDels_gram_3_stride_1_slice_1837498.npz'
    ngram = 3
    create_tfrecord_only_sequence_all(seq_file, tfrecord_file, ngram=ngram)








