# -*- coding:utf-8 -*-
import os
import sys

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Reshape, Permute, concatenate
from tensorflow.keras.layers import Lambda, Dense, Layer
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
import numpy
from sklearn import metrics

sys.path.append("../")
from bgi.bert4keras.models import build_transformer_model
from bgi.common.callbacks import LRSchedulerPerStep
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number


def load_npz_data_for_classification(file_name, ngram=3, only_one_slice=True, ngram_index=None, masked=True):
    """
    Import npz data
    :param file_name:
    :param ngram:
    :param only_one_slice:
    :param ngram_index:
    :return:
    """
    x_data_all = []
    anno_data_all = []
    y_data_all = []
    if str(file_name).endswith('.npz') is False or os.path.exists(file_name) is False:
        return x_data_all, None, y_data_all

    loaded = np.load(file_name)
    x_data = loaded['sequence']
    y_data = loaded['label']

    # if masked:
    #     positive_samples = np.sum(y_data)
    #     RANDOM_STATE = 42
    #     print("positive_samples: ", positive_samples)
    #     x_data, y_data = make_imbalance(x_data, y_data,
    #                                     sampling_strategy={0: positive_samples, 1: positive_samples},
    #                                     random_state=RANDOM_STATE)

    print("Load: ", file_name)
    print("X: ", x_data.shape)
    print("Y: ", y_data.shape)
    if only_one_slice is True:
        for ii in range(ngram):
            if ngram_index is not None and ii != ngram_index:
                continue
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

    return x_data_all, anno_data_all, y_data_all


def load_all_data(record_names: list, ngram=3, only_one_slice=True, ngram_index=None, masked=False):
    x_data_all = []
    y_data_all = []
    for file_name in record_names:
        x_data, anno_data, y_data = load_npz_data_for_classification(file_name,
                                                                     ngram,
                                                                     only_one_slice,
                                                                     ngram_index,
                                                                     masked=masked
                                                                     )
        x_data_all.extend(x_data)
        y_data_all.extend(y_data)

    x_data_all = np.concatenate(x_data_all)
    y_data_all = np.concatenate(y_data_all)
    return x_data_all, y_data_all


# @tf.function
def load_npz_dataset_for_classification(x_promoter_data_all,
                                        y_data_all,
                                        promoter_seq_len,
                                        ngram=5,
                                        only_one_slice=True,
                                        ngram_index=None,
                                        shuffle=False,
                                        seq_len=200,
                                        num_classes=1,
                                        masked=True,
                                        ):
    """
    Read sequence data from NPZ file and generate tf.data.DataSet
    :param record_names:
    :param batch_size:
    :param ngram:
    :param only_one_slice: Slice by ngram
    :param ngram_index:
    :param shuffle:
    :param seq_len:
    :param num_classes:
    :param num_parallel_calls:
    :return:
    """

    # if not isinstance(enhacer_record_names, list):
    #     enhacer_record_names = [enhacer_record_names]
    #
    # if not isinstance(promoter_record_names, list):
    #     promoter_record_names = [promoter_record_names]

    if num_classes == 1:
        y_data_all = np.reshape(y_data_all, (y_data_all.shape[0], 1))

    # Data Generator
    def data_generator():
        total_size = len(x_promoter_data_all)
        indexes = np.arange(total_size)
        if shuffle is True:
            np.random.shuffle(indexes)

        ii = 0
        while True:
            if ii < total_size:
                index = indexes[ii]
            else:
                print("Shuffle ..............................................")
                total_size = len(x_promoter_data_all)
                indexes = np.arange(total_size)
                if shuffle is True:
                    np.random.shuffle(indexes)

                ii = 0
                index = indexes[ii]

            x_promoter = x_promoter_data_all[index]

            segment_promoter = np.zeros_like(x_promoter)

            y = y_data_all[index]
            ii += 1
            yield x_promoter, segment_promoter, y

    classes_shape = tf.TensorShape([num_classes])
    if num_classes == 1:
        classes_shape = tf.TensorShape([1])

    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_types=(tf.int16, tf.int16, tf.int16),
                                             output_shapes=(
                                                 tf.TensorShape([promoter_seq_len]),
                                                 tf.TensorShape([promoter_seq_len]),
                                                 classes_shape
                                             ))
    return dataset


def parse_function(x_promoter, segment_x, y):
    x = {
        'Input-Token': x_promoter,
        'Input-Segment': segment_x,
    }

    y = {
        'CLS-Activation': y
    }
    return x, y


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, 'float')
    y_pred = tf.cast(y_pred, 'float')
    TP = K.sum(tf.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 1), 'float'))
    FP = K.sum(tf.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 1), 'float'))
    FN = K.sum(tf.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 0), 'float'))
    TN = K.sum(tf.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 0), 'float'))
    P = TP / (TP + FP + K.epsilon())
    R = TP / (TP + FN + K.epsilon())
    F1 = 2 * P * R / (P + R + K.epsilon())

    return F1


def auprc_score(y_true, y_pred):
    y_true = tf.cast(y_true, 'float')
    y_pred = tf.cast(y_pred, 'float')
    auprc = tf.py_function(metrics.average_precision_score, (y_true, y_pred), tf.float64)
    return auprc


def average_precision(y_true, y_pred):
    y_true = tf.cast(y_true, 'float')
    y_pred = tf.cast(y_pred, 'float')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def model_def(embedding_size=128,
              hidden_size=128,
              num_heads=8,
              num_hidden_layers=1,
              vocab_size=10000,
              drop_rate=0.25):
    config = {
        "attention_probs_dropout_prob": 0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0,
        "embedding_size": embedding_size,
        "hidden_size": hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "max_position_embeddings": 1024 * 2,
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_hidden_layers,
        "num_hidden_groups": 1,
        "net_structure_type": 0,
        "gap_size": 0,
        "num_memory_blocks": 0,
        "inner_group_num": 1,
        "down_scale_factor": 1,
        "type_vocab_size": 0,
        "vocab_size": vocab_size,
        "custom_masked_sequence": False,
        "custom_conv_layer": False,
        "use_segment_ids": True,
        "use_position_ids": True,
        "multi_inputs": []
    }
    bert = build_transformer_model(
        configs=config,
        model='bert',
        return_keras_model=False,
    )

    promoter_output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = BatchNormalization()(promoter_output)
    output = Dropout(drop_rate)(output)
    output = Dense(1, activation='sigmoid', name='CLS-Activation')(output)

    model = tf.keras.models.Model(inputs=bert.model.input, outputs=[output])

    return model


def train_kfold(train_data_file,
                data_path,
        batch_size=256,
                epochs=10,
                ngram=6,
                n_splits=10,
                vocab_size=10000,
                PROMOTER_RESIZED_LEN=600,
                task_name='epdnew_both'):
    # Distributed Training
    num_gpu = 1
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync >= 1:
        num_gpu = strategy.num_replicas_in_sync

    GLOBAL_BATCH_SIZE = batch_size * num_gpu
    num_parallel_calls = 16

    ## load data: sequence, label
    train_promoter_files = [os.path.join(data_path, train_data_file)]

    print("train_promoter_files: ", train_promoter_files)

    only_one_slice = True
    region1_seq, label = load_all_data(train_promoter_files, ngram=ngram, only_one_slice=only_one_slice,
                                       ngram_index=None)

    seed = 7
    numpy.random.seed(seed)
    X = region1_seq
    Y = label

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    promoter_seq_len = PROMOTER_RESIZED_LEN // ngram * ngram

    if only_one_slice:
        promoter_seq_len = promoter_seq_len // ngram

    k_fold = 0
    shuffle = True
    for train, test in kfold.split(X, Y):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)

        promoter_indexes = np.arange(len(train))
        if shuffle is True:
            np.random.shuffle(promoter_indexes)

        train_slice = promoter_indexes[:int(len(promoter_indexes) * 0.9)]
        valid_slice = promoter_indexes[int(len(promoter_indexes) * 0.9):]

        x_train_data = X[train[train_slice]]
        y_train_data = Y[train[train_slice]]

        x_valid_data = X[train[valid_slice]]
        y_valid_data = Y[train[valid_slice]]

        x_test_data = X[test]
        y_test_data = Y[test]


        np.savez('./data/kfold_{}_train_and_valid_index_{}_gram_{}.npz'.format(str(k_fold), str(ngram), task_name), train=train,
                 test=test)

        with strategy.scope():
            model = model_def(vocab_size=vocab_size)
            print('compiling...')
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(0.0001),
                          metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score])
            # model.summary()

        filename = './data/promoter_best_model_gene_bert_{}_fold_{}_gram_{}.h5'.format(str(k_fold), str(ngram), task_name)

        modelCheckpoint = ModelCheckpoint(filename, monitor='val_acc', save_best_only=True, verbose=1)

        print('fitting...')

        train_total_size = len(y_train_data)
        train_dataset = load_npz_dataset_for_classification(x_train_data,
                                                            y_train_data,
                                                            promoter_seq_len,
                                                            ngram=ngram,
                                                            only_one_slice=only_one_slice,
                                                            ngram_index=None,
                                                            shuffle=True,
                                                            seq_len=0,
                                                            masked=False,
                                                            )

        train_dataset = train_dataset.shuffle(train_total_size, reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        valid_total_size = len(y_valid_data)
        valid_dataset = load_npz_dataset_for_classification(x_valid_data,
                                                            y_valid_data,
                                                            promoter_seq_len,
                                                            ngram=ngram,
                                                            only_one_slice=only_one_slice,
                                                            ngram_index=None,
                                                            shuffle=False,
                                                            seq_len=0,
                                                            num_classes=1,
                                                            masked=False,
                                                            )
        valid_dataset = valid_dataset.batch(batch_size)
        valid_dataset = valid_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
        valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        test_total_size = len(y_test_data)
        test_dataset = load_npz_dataset_for_classification(x_test_data,
                                                            y_test_data,
                                                            promoter_seq_len,
                                                            ngram=ngram,
                                                            only_one_slice=only_one_slice,
                                                            ngram_index=None,
                                                            shuffle=False,
                                                            seq_len=0,
                                                            num_classes=1,
                                                            masked=False,
                                                            )
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)


        train_steps_per_epoch = train_total_size // GLOBAL_BATCH_SIZE
        valid_steps_per_epoch = valid_total_size // GLOBAL_BATCH_SIZE
        test_steps_per_epoch = test_total_size // GLOBAL_BATCH_SIZE

        print("Training")
        print("batch size: ", GLOBAL_BATCH_SIZE)

        model_train_history = model.fit(train_dataset,
                                        steps_per_epoch=train_steps_per_epoch,
                                        epochs=epochs,
                                        validation_data=valid_dataset,
                                        validation_steps=valid_steps_per_epoch,
                                        callbacks=[modelCheckpoint, early_stopping],
                                        verbose=2)

        print(model_train_history)

        # Make predictions and reload the optimal weights
        with strategy.scope():
            model = model_def(vocab_size=vocab_size)
            print('compiling...')
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(0.0001),
                          metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score])
            model.load_weights(filename)
        eval = model.evaluate(test_dataset, steps=test_steps_per_epoch, verbose=2)
        print("Eval: ", eval)

        k_fold += 1


if __name__ == '__main__':

    # Dynamic allocation of video memory
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    ngram = 3
    stride = 1
    word_dict = get_word_dict_for_n_gram_number(n_gram=ngram)
    vocab_size = len(word_dict) + 10

    # train_kfold(batch_size=128, epochs=20, vocab_size=vocab_size, task_name='epdnew_NO_TATA_BOX')

    # evaluate(CELL, TYPE, NUM_ENSEMBL=10)

    # BOTH
    train_data_file = 'epdnew_BOTH_Knowledge_{}_gram.npz'.format(str(ngram))
    data_path = './data/' + '{}_gram'.format(ngram)
    task_name = 'epdnew_BOTH'

    annotation_size = 11
    train_kfold(train_data_file=train_data_file,
                data_path=data_path,
                batch_size=256,
                epochs=20,
                vocab_size=vocab_size,
                task_name=task_name)

    # TATA BOX
    train_data_file = 'epdnew_TATA_BOX_Knowledge_{}_gram.npz'.format(str(ngram))
    data_path = './data/' + '{}_gram'.format(ngram)
    task_name = 'epdnew_TATA_BOX'

    annotation_size = 11
    train_kfold(train_data_file=train_data_file,
                data_path=data_path,
                batch_size=256,
                epochs=20,
                vocab_size=vocab_size,
                task_name=task_name)

    # NO TATA BOX
    train_data_file = 'epdnew_NO_TATA_BOX_Knowledge_{}_gram.npz'.format(str(ngram))
    data_path = './data/' + '{}_gram'.format(ngram)
    task_name = 'epdnew_NO_TATA_BOX'

    annotation_size = 11
    train_kfold(train_data_file=train_data_file,
                data_path=data_path,
                batch_size=256,
                epochs=20,
                vocab_size=vocab_size,
                task_name=task_name)
