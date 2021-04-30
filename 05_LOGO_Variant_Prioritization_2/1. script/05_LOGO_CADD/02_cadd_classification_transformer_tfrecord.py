#! -*- coding:utf-8 -*-

import json
import os
import random
import sys
import h5py

import numpy as np
import tensorflow as tf
# from keras.callbacks import ModelCheckpoint
# from keras.layers import Dense, BatchNormalization, Activation, Embedding, GlobalAveragePooling1D, Input
# from keras.datasets import imdb

from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tqdm import tqdm
from sklearn import metrics

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda, Dense

sys.path.append("../../")
from bgi.bert4keras.models import build_transformer_model
from bgi.common.refseq_utils import get_word_dict_for_n_gram_alphabet


def single_file_dataset(input_file, seq_len=50):
    d = tf.data.TFRecordDataset(input_file)

    def single_example_parser(serialized_example):
        name_to_features = {
            # 'feature': tf.io.VarLenFeature(tf.int64),
            # 'weight': tf.io.VarLenFeature(tf.float32),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'seq': tf.io.FixedLenFeature([seq_len], tf.int64),
            'alt_seq': tf.io.FixedLenFeature([seq_len], tf.int64),
            'alt_type': tf.io.FixedLenFeature([seq_len], tf.int64),
        }
        example = tf.io.parse_single_example(serialized_example, name_to_features)

        # feature = example['feature']
        # weight = example['weight']
        label = example['label']
        ref_seq = example['seq']
        alt_seq = example['alt_seq']
        alt_type = example['alt_type']

        # feature = tf.sparse.to_dense(feature, default_value=0)
        # weight = tf.sparse.to_dense(weight, default_value=0)

        return ref_seq, alt_seq, alt_type, label

    d = d.map(single_example_parser)

    return d


def create_classifier_dataset(file_names,
                              batch_size,
                              epochs=10,
                              seq_len=50,
                              is_training=True,
                              shuffle_size=100):
    # 读取记录
    dataset = single_file_dataset(file_names)

    def _select_data_from_record(ref_seq, alt_seq, alt_type, label):
        x = {
            # 'Input-Token': feature,
            # 'Input-Weight': weight,
            'Input-Token-ALT': ref_seq,
            'Input-Token-Alt-ALT': alt_seq,
            'Input-Segment-ALT': alt_type,
        }
        y = label
        return x, y

    dataset = dataset.map(_select_data_from_record, num_parallel_calls=16)
    if is_training:
        dataset = dataset.shuffle(shuffle_size)

    dataset = dataset.batch(batch_size)

    return dataset


def get_model(max_features=1000,
              embedding_dims=128,
              model_dim=128,
              num_heads=8,
              hidden_layers=1,
              num_classes=1,
              activation='sigmoid',
              vocab_size=10000,
              ):
    x_in = Input(shape=(None,), name='Input-Token')
    x_weight = Input(shape=(None,), name='Input-Weight', dtype='float')
    x = Embedding(max_features, embedding_dims, input_length=None)(x_in)
    sparse_value = tf.expand_dims(x_weight, 2)
    sparse_value = BatchNormalization()(sparse_value)
    x = tf.multiply(x, sparse_value)

    # First order
    first_order = GlobalAveragePooling1D()(x)
    first_order = BatchNormalization()(first_order)

    # Second order
    # x1 = x
    # for ii in range(0):
    #     xv = Dense(embedding_dims)(x)
    #     xw = Dense(embedding_dims)(x)
    #     xn = Dense(embedding_dims)(x)
    #     square_inputs = K.square(K.sum(xv, axis=-1))
    #     sum_inputs = K.sum(K.square(xw), axis=-1)
    #     second_order = square_inputs - sum_inputs
    #     second_order = tf.nn.softmax(second_order)
    #     sparse_value = tf.expand_dims(second_order, 2)
    #     second_order = tf.multiply(sparse_value, xn)
    #     # second_order = tf.multiply(second_order, mask_value)
    #     second_order = tf.keras.layers.add([second_order, x1])
    #     x = second_order
    # second_order = x
    # second_order = GlobalAveragePooling1D()(second_order)

    # Transformer模型配置
    config = {
        "attention_probs_dropout_prob": 0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0,
        "embedding_size": embedding_dims,
        "hidden_size": model_dim,
        "initializer_range": 0.02,
        "intermediate_size": model_dim * 4,
        "max_position_embeddings": 512,
        "num_attention_heads": num_heads,
        "num_hidden_layers": hidden_layers,
        "num_hidden_groups": 1,
        "net_structure_type": 0,
        "gap_size": 0,
        "num_memory_blocks": 0,
        "inner_group_num": 1,
        "down_scale_factor": 1,
        "type_vocab_size": 0,
        "vocab_size": vocab_size,
        "custom_masked_sequence": False,
        "multi_inputs": [],
        "custom_conv_layer": True,
        "use_segment_ids": True,
        "use_position_ids": True,
    }
    seq_bert = build_transformer_model(
        configs=config,
        # checkpoint_path=checkpoint_path,
        model='multi_inputs_alt_bert',
        return_keras_model=False,
    )
    seq_feature = Lambda(lambda x: x[:, 0], name='CLS-token-Seq')(seq_bert.model.output)
    # seq_feature = tf.reduce_mean(seq_bert.model.output, axis=-1)
    # seq_feature = tf.keras.layers.GlobalMaxPool1D()(seq_bert.model.output)

    print("seq_feature: ", seq_feature)
    print("first_order: ", first_order)
    inputs = []
    # inputs.extend([x_in, x_weight])
    inputs.extend(seq_bert.model.input)
    # Concatenate
    feature = seq_feature  # concatenate([first_order, seq_feature])
    output = Dense(num_classes, activation=activation, use_bias=True, name='CLS-Activation')(feature)
    model = Model(inputs, output)

    return model


if __name__ == '__main__':

    # 动态分配显存
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    num_classes = 1
    maxlen = 1000
    batch_size = 1024
    num_gpu = 1
    vocab_size = 1000

    epochs = 100

    ngram = 3
    seq_size = 50
    word_index_from = 10
    word_dict = get_word_dict_for_n_gram_alphabet(n_gram=ngram, word_index_from=word_index_from)
    vocab_size = len(word_dict) + word_index_from

    task = 'eval'

    nfolds = 5
    if task == 'train':

        for fold in range(nfolds):
            # 模型保存
            save_path = './models/'
            model_name = 'cadd_indels_seq_noncoding'
            filepath = os.path.join(save_path, model_name + "_classification_{}_fold.hdf5".format(str(fold)))
            loss_name = "accuracy"
            mc = ModelCheckpoint(filepath, monitor=loss_name, save_best_only=True, verbose=1)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

            model = get_model(num_classes=num_classes,
                              embedding_dims=128,
                              hidden_layers=1,
                              vocab_size=vocab_size,
                              activation='sigmoid')
            model.summary()
            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                          optimizer=Adam(2e-6),
                          metrics=['accuracy', tf.keras.metrics.AUC()])

            train_slice_files = [
                '/alldata/Hphuang_data/Genomics/CADD/GRCh37/humanDerived_InDels_seq_train_3_gram.tfrecord',
                '/alldata/Hphuang_data/Genomics/CADD/GRCh37/simulation_InDels_seq_train_3_gram.tfrecord',
                #
                # 'D:\\Research\\Data\\Genomics\\CADD\\GRCh37\\humanDerived_InDels_seq_train_3_gram.tfrecord',
                # 'D:\\Research\\Data\\Genomics\\CADD\\GRCh37\\simulation_InDels_seq_train_3_gram.tfrecord',
            ]
            valid_slice_files = [
                './data/fold{}/clinvar_InDel_fold{}_valid_gram_3_stride_1_slice_2934.tfrecord'.format(fold, fold)
            ]
            test_slice_files = [
                './data/fold{}/clinvar_InDel_fold{}_test_gram_3_stride_1_slice_2935.tfrecord'.format(fold, fold)
            ]

            train_total_size = 15000000 * 0.9
            valid_total_size = 15000000 * 0.05
            test_total_size = 15000000 * 0.05

            GLOBAL_BATCH_SIZE = batch_size * num_gpu
            train_steps_per_epoch = int(train_total_size // GLOBAL_BATCH_SIZE)
            valid_steps_per_epoch = int(valid_total_size // 2 // GLOBAL_BATCH_SIZE)
            test_steps_per_epoch = int(valid_total_size // 2 // GLOBAL_BATCH_SIZE)


            train_dataset = create_classifier_dataset(train_slice_files,
                                                      batch_size=batch_size,
                                                      is_training=True,
                                                      epochs=epochs,
                                                      shuffle_size=train_total_size
                                                      )
            train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
            train_dataset = train_dataset.repeat()

            valid_dataset = create_classifier_dataset(valid_slice_files,
                                                      batch_size=batch_size,
                                                      is_training=False,
                                                      epochs=epochs,
                                                      shuffle_size=valid_total_size
                                                      )
            valid_dataset = valid_dataset.prefetch(valid_steps_per_epoch)
            valid_dataset = valid_dataset.repeat()

            test_dataset = create_classifier_dataset(test_slice_files,
                                                     batch_size=batch_size,
                                                     is_training=False,
                                                     epochs=epochs,
                                                     shuffle_size=test_total_size
                                                     )
            test_dataset = test_dataset.prefetch(test_steps_per_epoch)
            test_dataset = test_dataset.repeat()

            model_train_history = model.fit(train_dataset,
                                            steps_per_epoch=train_steps_per_epoch,
                                            epochs=epochs,
                                            validation_data=valid_dataset,
                                            validation_steps=valid_steps_per_epoch,
                                            callbacks=[mc, early_stopping],
                                            verbose=2)

            eval_model = get_model(num_classes=num_classes,
                                   embedding_dims=128,
                                   hidden_layers=1,
                                   vocab_size=vocab_size,
                                   activation='sigmoid')
            eval_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                          optimizer=Adam(2e-6),
                          metrics=['accuracy', tf.keras.metrics.AUC()])
            eval_model.load_weights(filepath)
            eval = eval_model.evaluate(test_dataset, steps=test_steps_per_epoch)
            print("eval: ", eval)


    if task == 'eval':
        for fold in range(nfolds):
            # 模型保存
            save_path = './models/'
            model_name = 'cadd_indels_seq_noncoding'
            filepath = os.path.join(save_path, model_name + "_classification_{}_fold.hdf5".format(str(fold)))
            loss_name = "accuracy"
            mc = ModelCheckpoint(filepath, monitor=loss_name, save_best_only=True, verbose=1)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

            model = get_model(num_classes=num_classes,
                              embedding_dims=128,
                              hidden_layers=1,
                              vocab_size=vocab_size,
                              activation='sigmoid')
            model.summary()
            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                          optimizer=Adam(2e-6),
                          metrics=['accuracy', tf.keras.metrics.AUC()])

            train_slice_files = [
                '/alldata/Hphuang_data/Genomics/CADD/GRCh37/humanDerived_InDels_seq_train_3_gram.tfrecord',
                '/alldata/Hphuang_data/Genomics/CADD/GRCh37/simulation_InDels_seq_train_3_gram.tfrecord',
                #
                # 'D:\\Research\\Data\\Genomics\\CADD\\GRCh37\\humanDerived_InDels_seq_train_3_gram.tfrecord',
                # 'D:\\Research\\Data\\Genomics\\CADD\\GRCh37\\simulation_InDels_seq_train_3_gram.tfrecord',
            ]
            valid_slice_files = [
                './data/fold{}/clinvar_InDel_fold{}_valid_gram_3_stride_1_slice_2934.tfrecord'.format(fold, fold)
            ]
            test_slice_files = [
                './data/fold{}/clinvar_InDel_fold{}_test_gram_3_stride_1_slice_2935.tfrecord'.format(fold, fold)
            ]

            train_total_size = 1800000 * 2
            valid_total_size = 5869
            test_total_size = 5869

            GLOBAL_BATCH_SIZE = batch_size * num_gpu
            train_steps_per_epoch = int(train_total_size // GLOBAL_BATCH_SIZE)
            valid_steps_per_epoch = int(valid_total_size // 2 // GLOBAL_BATCH_SIZE)
            test_steps_per_epoch = int(valid_total_size // 2 // GLOBAL_BATCH_SIZE)

            # train_dataset = create_classifier_dataset(train_slice_files,
            #                                           batch_size=batch_size,
            #                                           is_training=True,
            #                                           epochs=epochs,
            #                                           shuffle_size=train_total_size
            #                                           )
            # train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
            # train_dataset = train_dataset.repeat()
            #
            # valid_dataset = create_classifier_dataset(valid_slice_files,
            #                                           batch_size=batch_size,
            #                                           is_training=False,
            #                                           epochs=epochs,
            #                                           shuffle_size=valid_total_size
            #                                           )
            # valid_dataset = valid_dataset.prefetch(valid_steps_per_epoch)
            # valid_dataset = valid_dataset.repeat()

            test_dataset = create_classifier_dataset(test_slice_files,
                                                     batch_size=batch_size,
                                                     is_training=False,
                                                     epochs=epochs,
                                                     shuffle_size=test_total_size
                                                     )
            test_dataset = test_dataset.prefetch(test_steps_per_epoch)
            test_dataset = test_dataset.repeat()

            # model_train_history = model.fit(train_dataset,
            #                                 steps_per_epoch=train_steps_per_epoch,
            #                                 epochs=epochs,
            #                                 validation_data=valid_dataset,
            #                                 validation_steps=valid_steps_per_epoch,
            #                                 callbacks=[mc, early_stopping],
            #                                 verbose=2)

            # eval_model = get_model(num_classes=num_classes,
            #                        embedding_dims=128,
            #                        hidden_layers=1,
            #                        vocab_size=vocab_size,
            #                        activation='sigmoid')
            # eval_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            #                    optimizer=Adam(2e-6),
            #                    metrics=['accuracy', tf.keras.metrics.AUC()])
            model.load_weights(filepath)
            eval = model.evaluate(test_dataset, steps=test_steps_per_epoch)
            print("eval: ", eval)

            preds = model.predict(test_dataset, steps=test_steps_per_epoch)
            print("preds: ", preds)
