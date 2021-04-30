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
    x_data = loaded['x']
    y_data = loaded['y']

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
def load_npz_dataset_for_classification(x_enhancer_data_all,
                                        x_promoter_data_all,
                                        y_data_all,
                                        enhancer_seq_len,
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
        total_size = len(x_enhancer_data_all)
        indexes = np.arange(total_size)
        if shuffle is True:
            np.random.shuffle(indexes)

        ii = 0
        while True:
            if ii < total_size:
                index = indexes[ii]
            else:
                print("Shuffle ..............................................")
                total_size = len(x_enhancer_data_all)
                indexes = np.arange(total_size)
                if shuffle is True:
                    np.random.shuffle(indexes)

                ii = 0
                index = indexes[ii]

            x_enhancer = x_enhancer_data_all[index]
            x_promoter = x_promoter_data_all[index]

            segment_promoter = np.zeros_like(x_promoter)
            segment_enhancer = np.zeros_like(x_enhancer)

            y = y_data_all[index]
            ii += 1
            yield x_enhancer, segment_enhancer, x_promoter, segment_promoter, y

    classes_shape = tf.TensorShape([num_classes])
    if num_classes == 1:
        classes_shape = tf.TensorShape([1])

    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_types=(tf.int16, tf.int16, tf.int16, tf.int16, tf.int16),
                                             output_shapes=(
                                                 tf.TensorShape([enhancer_seq_len]),
                                                 tf.TensorShape([enhancer_seq_len]),
                                                 tf.TensorShape([promoter_seq_len]),
                                                 tf.TensorShape([promoter_seq_len]),
                                                 classes_shape
                                             ))
    return dataset


def parse_function(x_enhancer, segment_enhancer, x_promoter, segment_x, y):
    x = {
        'Input-Token_1': x_enhancer,
        'Input-Segment_1': segment_enhancer,
        'Input-Token_2': x_promoter,
        'Input-Segment_2': segment_x,
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
              drop_rate=0.25,
              ENHANCER_RESIZED_LEN=2000,
              PROMOTER_RESIZED_LEN=1000):
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
        "custom_conv_layer": True,
        "use_segment_ids": True,
        "use_position_ids": True,
        "multi_inputs": []
    }
    bert1 = build_transformer_model(
        configs=config,
        model='bert',
        return_keras_model=False,
    )

    bert2 = build_transformer_model(
        configs=config,
        model='bert',
        return_keras_model=False,
    )

    x_promoter = tf.keras.layers.Input(shape=(None,), name='Input-Token_1')
    s_promoter = tf.keras.layers.Input(shape=(None,), name='Input-Segment_1')
    x_enhancer = tf.keras.layers.Input(shape=(None,), name='Input-Token_2')
    s_segment = tf.keras.layers.Input(shape=(None,), name='Input-Segment_2')

    promoter_output = bert1.model([x_promoter, s_promoter])
    enhancer_output = bert2.model([x_enhancer, s_segment])

    promoter_output = Lambda(lambda x: x[:, 0])(promoter_output)
    enhancer_output = Lambda(lambda x: x[:, 0])(enhancer_output)

    x = concatenate([promoter_output, enhancer_output])
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)
    # x = Bidirectional(LSTM(128, return_sequences=False), merge_mode='concat')(x)
    # x = Activation('relu')(x)
    # x = Dropout(drop_rate)(x)
    # x = tf.keras.layers.Flatten()(x)
    output = Dense(1, activation='sigmoid', name='CLS-Activation')(x)

    model = tf.keras.models.Model([x_promoter, s_promoter, x_enhancer, s_segment], output)

    return model


def train_kfold(CELL, TYPE,
                batch_size=256,
                epochs=10,
                ngram=6,
                NUM_SEQ=4,
                vocab_size=10000,
                ENHANCER_RESIZED_LEN=2000,
                PROMOTER_RESIZED_LEN=1000):
    # Distributed Training
    num_gpu = 1
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync >= 1:
        num_gpu = strategy.num_replicas_in_sync

    GLOBAL_BATCH_SIZE = batch_size * num_gpu
    num_parallel_calls = 16

    ## load data: sequence
    data_path = CELL + '/' + TYPE + '/' + '{}_gram'.format(ngram)
    train_enhancer_files = [data_path + '/enhancer_Seq_{}_gram.npz'.format(ngram)]
    train_promoter_files = [data_path + '/promoter_Seq_{}_gram.npz'.format(ngram)]

    print("train_enhancer_files: ", train_enhancer_files)
    print("train_promoter_files: ", train_promoter_files)

    region1_seq, label = load_all_data(train_enhancer_files, ngram=ngram, only_one_slice=True,
                                       ngram_index=1)
    region2_seq, _ = load_all_data(train_promoter_files, ngram=ngram, only_one_slice=True,
                                   ngram_index=1)

    seed = 7
    numpy.random.seed(seed)
    X = np.hstack([region1_seq, region2_seq])
    Y = label

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    enhancer_seq_len = ENHANCER_RESIZED_LEN // ngram // ngram * ngram
    promoter_seq_len = PROMOTER_RESIZED_LEN // ngram // ngram * ngram

    k_fold = 0
    for train, test in kfold.split(X, Y):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)

        x_train_data = X[train]
        y_train_data = Y[train]

        x_valid_data = X[test]
        y_valid_data = Y[test]

        np.savez(CELL + '/' + TYPE + '/kfold_{}_train_and_valid_index.npz'.format(str(k_fold)), train=train, test=test)

        with strategy.scope():
            model = model_def(vocab_size=vocab_size)
            print('compiling...')
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(lr=0.00001),
                          metrics=['acc', f1_score, tf.keras.metrics.AUC()])

        filename = CELL + '/' + TYPE + '/best_model_gene_bert_{}.h5'.format(str(k_fold))

        modelCheckpoint = ModelCheckpoint(filename, monitor='val_acc', save_best_only=True, verbose=1)

        print('fitting...')
        region1_seq = x_train_data[:, 0:enhancer_seq_len]
        region2_seq = x_train_data[:, enhancer_seq_len:]

        region1_seq_valid = x_valid_data[:, 0:enhancer_seq_len]
        region2_seq_valid = x_valid_data[:, enhancer_seq_len:]

        train_total_size = len(y_train_data)
        train_dataset = load_npz_dataset_for_classification(region1_seq,
                                                            region2_seq,
                                                            y_train_data,
                                                            enhancer_seq_len,
                                                            promoter_seq_len,
                                                            ngram=ngram,
                                                            only_one_slice=True,
                                                            ngram_index=1,
                                                            shuffle=True,
                                                            seq_len=0,
                                                            masked=False,
                                                            )

        train_dataset = train_dataset.shuffle(train_total_size, reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        valid_total_size = len(y_valid_data)
        valid_dataset = load_npz_dataset_for_classification(region1_seq_valid,
                                                            region2_seq_valid,
                                                            y_valid_data,
                                                            enhancer_seq_len,
                                                            promoter_seq_len,
                                                            ngram=ngram,
                                                            only_one_slice=True,
                                                            ngram_index=1,
                                                            shuffle=False,
                                                            seq_len=0,
                                                            num_classes=1,
                                                            masked=False,
                                                            )
        valid_dataset = valid_dataset.batch(batch_size)
        valid_dataset = valid_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
        valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        train_steps_per_epoch = train_total_size // GLOBAL_BATCH_SIZE
        valid_steps_per_epoch = valid_total_size // GLOBAL_BATCH_SIZE

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
        k_fold += 1


def bagging_predict(label, bag_pred, bag_score):
    vote_pred = np.zeros(bag_pred.shape[1])
    vote_score = np.zeros(bag_score.shape[1])
    import scipy.stats as stats
    for i in range(bag_pred.shape[1]):
        vote_pred[i] = stats.mode(bag_pred[:, i]).mode
        vote_score[i] = np.mean(bag_score[:, i])
    f1 = metrics.f1_score(label, vote_pred)
    auprc = metrics.average_precision_score(label, vote_score)
    return f1, auprc

# @tf.function
def evaluate(CELL, TYPE, NUM_ENSEMBL=1, ngram=6, batch_size=128, num_parallel_calls=16, ENHANCER_RESIZED_LEN=2000, PROMOTER_RESIZED_LEN=1000):
    ## load data: sequence
    data_path = CELL + '/' + TYPE + '/test/'
    ## load data: sequence
    data_path = CELL + '/' + TYPE + '/test/' + '{}_gram'.format(ngram)
    test_enhancer_files = [data_path + '/enhancer_Seq_{}_gram.npz'.format(ngram)]
    test_promoter_files = [data_path + '/promoter_Seq_{}_gram.npz'.format(ngram)]

    print("test_enhancer_files: ", test_enhancer_files)
    print("test_promoter_files: ", test_promoter_files)

    region1_seq, label = load_all_data(test_enhancer_files, ngram=ngram, only_one_slice=True,
                                       ngram_index=1)
    region2_seq, _ = load_all_data(test_promoter_files, ngram=ngram, only_one_slice=True,
                                   ngram_index=1)

    bag_pred = np.zeros((NUM_ENSEMBL, label.shape[0]))
    bag_score = np.zeros((NUM_ENSEMBL, label.shape[0]))

    epochs = 1
    for t in range(NUM_ENSEMBL):
        model = model_def(vocab_size=vocab_size)
        print('compiling...')
        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=0.00001),
                      metrics=['acc', f1_score, tf.keras.metrics.AUC(), average_precision])

        weight_path = CELL + '/' + TYPE + '/best_model_gene_bert_' + str(t) + '_epoch_05' +  '.h5'
        if os.path.exists(weight_path):
            model.load_weights(weight_path)
            print("Loading: ", weight_path)

        enhancer_seq_len = ENHANCER_RESIZED_LEN // ngram // ngram * ngram
        promoter_seq_len = PROMOTER_RESIZED_LEN // ngram // ngram * ngram
        valid_total_size = len(label)
        valid_dataset = load_npz_dataset_for_classification(region1_seq,
                                                            region2_seq,
                                                            label,
                                                            enhancer_seq_len,
                                                            promoter_seq_len,
                                                            ngram=ngram,
                                                            only_one_slice=True,
                                                            ngram_index=1,
                                                            shuffle=False,
                                                            seq_len=0,
                                                            num_classes=1,
                                                            masked=False,
                                                            )
        # valid_dataset = valid_dataset.shuffle(len(label), reshuffle_each_iteration=True)
        valid_dataset = valid_dataset.batch(batch_size)
        valid_dataset = valid_dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
        valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        steps_per_epoch = valid_total_size // batch_size + 1
        eval = model.evaluate(valid_dataset, steps=steps_per_epoch, verbose=2)
        print("eval: ", eval)

        score = model.predict(valid_dataset, steps=steps_per_epoch)
        score = score[:len(label)]
        bag_pred[t, :] = (score > 0.5).astype(int).reshape(-1)
        bag_score[t, :] = score.reshape(-1)


    f1, auprc = bagging_predict(label, bag_pred, bag_score)
    print(f1, auprc)
    print("AUPRC: ", auprc)
    print("F1: ", f1)



if __name__ == '__main__':

    # Dynamic allocation of video memory
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    ngram = 6
    stride = 1
    word_dict = get_word_dict_for_n_gram_number(n_gram=ngram)
    vocab_size = len(word_dict) + 10

    CELLs = ['tB', 'FoeT', 'Mon', 'nCD4', 'tCD4', 'tCD8']
    TYPE = 'P-E'
    for CELL in CELLs:
        train_kfold(CELL, TYPE, batch_size=128, epochs=10, vocab_size=vocab_size)


