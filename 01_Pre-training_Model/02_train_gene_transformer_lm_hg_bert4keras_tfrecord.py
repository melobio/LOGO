# -*- coding:utf-8 -*-

import argparse
import json
import os
import random
from multiprocessing import Pool
import gc

import numpy as np
import tensorflow as tf

import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

import sys
sys.path.append("../")
from bgi.bert4keras.backend import K
from bgi.bert4keras.models import build_transformer_model
from bgi.common.callbacks import LRSchedulerPerStep
from bgi.common.refseq_utils import get_word_dict_for_n_gram_alphabet



# from bgi.transformers.configuration_albert import AlbertConfig
# from bgi.transformers.modeling_tf_albert import TFAlbertModel, TFAlbertMLMHead

floatx = K.floatx()
os.environ.setdefault('TF_KERAS', '1')


special_symbols = {
    "<unk>": 0,
    "<cls>": 1,
    "<mask>": 2,
}

VOCAB_SIZE = 32000
CLS_ID = special_symbols["<cls>"]
MASK_ID = special_symbols["<mask>"]



def load_tfrecord(record_names, sequence_length=100, num_parallel_calls=tf.data.experimental.AUTOTUNE, batch_size=32):
    """
    parse_function
    """
    def parse_function(serialized):
        features = {
            'masked_sequence': tf.io.FixedLenFeature([sequence_length], tf.int64),
            # 'segmentId': tf.io.FixedLenFeature([sequence_length], tf.int64),
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

        y = K.cast(sequence, K.floatx())
        y = K.reshape(y, (y.shape[0], 1))

        y = {
            ' MLM-Activation': y
        }
        # print("y: ", y.shape)
        return x, y

    if not isinstance(record_names, list):
        record_names = [record_names]

    dataset = tf.data.TFRecordDataset(record_names)

    dataset = dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
    # dataset = dataset.repeat()
    # dataset = dataset.shuffle(batch_size * 1000)
    # dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset



if __name__ == '__main__':

    _argparser = argparse.ArgumentParser(
        description='A simple example of the Transformer language model in Genomics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--save', type=str, required=True, metavar='PATH',
        help='A path where the best model should be saved / restored from')
    _argparser.add_argument(
        '--model-name', type=str, default='transformer_gene', metavar='NAME',
        help='The name of the saving model')
    # _argparser.add_argument(
    #     '--tensorboard-log', type=str, metavar='PATH', default=None,
    #     help='Path to a directory for Tensorboard logs')
    _argparser.add_argument(
        '--train-data', type=str, metavar='PATH', default=None,
        help='Path to a file of train data')
    _argparser.add_argument(
        '--valid-data', type=str, metavar='PATH', default=None,
        help='Path to a file of valid data')
    _argparser.add_argument(
        '--test-data', type=str, metavar='PATH', default=None,
        help='Path to a file of test data')
    _argparser.add_argument(
        '--epochs', type=int, default=150, metavar='INTEGER',
        help='The number of epochs to train')
    _argparser.add_argument(
        '--lr', type=float, default=2e-4, metavar='FLOAT',
        help='Learning rate')
    _argparser.add_argument(
        '--batch-size', type=int, default=32, metavar='INTEGER',
        help='Training batch size')
    _argparser.add_argument(
        '--seq-len', type=int, default=256, metavar='INTEGER',
        help='Max sequence length')
    _argparser.add_argument(
        '--we-size', type=int, default=128, metavar='INTEGER',
        help='Word embedding size')
    _argparser.add_argument(
        '--model', type=str, default='universal', metavar='NAME',
        choices=['universal', 'vanilla'],
        help='The type of the model to train: "vanilla" or "universal"')
    _argparser.add_argument(
        '--CUDA-VISIBLE-DEVICES', type=int, default=0, metavar='INTEGER',
        help='CUDA_VISIBLE_DEVICES')
    _argparser.add_argument(
        '--num-classes', type=int, default=10, metavar='INTEGER',
        help='Number of total classes')
    _argparser.add_argument(
        '--vocab-size', type=int, default=20000, metavar='INTEGER',
        help='Number of vocab')
    _argparser.add_argument(
        '--slice', type=list, default=[6],
        help='Slice')
    _argparser.add_argument(
        '--ngram', type=int, default=6, metavar='INTEGER',
        help='length of char ngram')
    _argparser.add_argument(
        '--stride', type=int, default=2, metavar='INTEGER',
        help='stride size')
    _argparser.add_argument(
        '--has-segment', action='store_true',
        help='Include segment ID')
    _argparser.add_argument(
        '--word-prediction', action='store_true',
        help='Word prediction')
    _argparser.add_argument(
        '--class-prediction', action='store_true',
        help='class prediction')
    _argparser.add_argument(
        '--num-heads', type=int, default=4, metavar='INTEGER',
        help='Heads of self attention')
    _argparser.add_argument(
        '--model-dim', type=int, default=128, metavar='INTEGER',
        help='Heads of self attention')
    _argparser.add_argument(
        '--transformer-depth', type=int, default=2, metavar='INTEGER',
        help='Heads of self attention')
    _argparser.add_argument(
        '--num-gpu', type=int, default=1, metavar='INTEGER',
        help='Number of GPUs')
    _argparser.add_argument(
        '--verbose', type=int, default=2, metavar='INTEGER',
        help='Verbose')
    _argparser.add_argument(
        '--weight-path', type=str, metavar='PATH', default=None,
        help='Path to a pretain weight')
    _argparser.add_argument(
        '--slice-counter', type=int, default=10, metavar='INTEGER',
        help='Heads of self attention')
    _argparser.add_argument(
        '--config-path', type=str, metavar='PATH', default=None,
        help='Path to configuration')
    _argparser.add_argument(
        '--pool-size', type=int, default=8, metavar='INTEGER',
        help='Pool size of multiprocessing')
    _argparser.add_argument(
        '--steps-per-epoch', type=int, default=10000, metavar='INTEGER',
        help='steps per epoch')
    _argparser.add_argument(
        '--shuffle-size', type=int, default=1000, metavar='INTEGER',
        help='Buffer shuffle size')
    _argparser.add_argument(
        '--num-parallel-calls', type=int, default=16, metavar='INTEGER',
        help='Num parallel calls')
    _argparser.add_argument(
        '--prefetch-buffer-size', type=int, default=4, metavar='INTEGER',
        help='Prefetch buffer size')

    _args = _argparser.parse_args()

    config_path = _args.config_path

    # Model
    model_name = _args.model_name
    save_path = _args.save
    config_save_path = os.path.join(save_path, "{}_config.json".format(model_name))

    # Batch size, Epochs, GPU
    batch_size = _args.batch_size
    epochs = _args.epochs
    num_gpu = _args.num_gpu

    verbose = _args.verbose

    # Max length of sequence
    max_seq_len = _args.seq_len


    # Sequence segmentation
    ngram = _args.ngram
    stride = _args.stride

    # Sequence segmentation dictionary
    word_from_index = 3
    word_dict = get_word_dict_for_n_gram_alphabet(n_gram=ngram)

    num_classes = _args.num_classes
    only_one_slice = True
    vocab_size = len(word_dict) + word_from_index + 10

    # Number of training files read each time
    slice_files = []
    slice_counter = _args.slice_counter

    # Read training file
    train_data_path = _args.train_data
    files = os.listdir(train_data_path)
    for file_name in files:
        if str(file_name).endswith('.tfrecord'):
            slice_files.append( os.path.join(train_data_path, file_name))

    # Model parameters
    max_depth = _args.transformer_depth
    model_dim = _args.model_dim
    embedding_size = _args.we_size
    num_heads = _args.num_heads
    class_prediction = _args.class_prediction
    word_prediction = _args.word_prediction
    word_seq_len =  max_seq_len // ngram * int(ngram/stride)

    print("max_seq_len: ", max_seq_len, " word_seq_len: ", word_seq_len)
    print("vocab_size: ", vocab_size)

    pool_size = _args.pool_size

    # Distributed Training
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync >= 1:
        num_gpu = strategy.num_replicas_in_sync

    with strategy.scope():
        # Model parameters
        config = {
            "attention_probs_dropout_prob": 0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0,
            "embedding_size": embedding_size,
            "hidden_size": model_dim,
            "initializer_range": 0.02,
            "intermediate_size": model_dim * 4,
            "max_position_embeddings": 512,
            "num_attention_heads": num_heads,
            "num_hidden_layers": max_depth,
            "num_hidden_groups": 1,
            "net_structure_type": 0,
            "gap_size": 0,
            "num_memory_blocks": 0,
            "inner_group_num": 1,
            "down_scale_factor": 1,
            "type_vocab_size": 0,
            "vocab_size": vocab_size,
            "custom_masked_sequence": False,
        }

        bert = build_transformer_model(
            configs=config,
            model='bert',
            with_mlm='linear',
            application='lm',
            return_keras_model=False
        )
        albert = bert.model
        albert.summary()
        albert.compile(optimizer='adam', loss=[tf.keras.losses.SparseCategoricalCrossentropy()], metrics=['accuracy'])

    with strategy.scope():
        pretrain_weight_path = _args.weight_path
        if pretrain_weight_path is not None and len(pretrain_weight_path) > 0:
            albert.load_weights(pretrain_weight_path, by_name=True)
            print("Load weights: ", pretrain_weight_path)

    # CallBack
    shuffle = True
    initial_epoch = 0
    steps_per_epoch = _args.steps_per_epoch
    shuffle_size = _args.shuffle_size
    num_parallel_calls = _args.num_parallel_calls
    prefetch_buffer_size = _args.prefetch_buffer_size

    lr_scheduler = LRSchedulerPerStep(model_dim,
                                      warmup=2500,
                                      initial_epoch=initial_epoch,
                                      steps_per_epoch=steps_per_epoch)

    loss_name = "loss"
    filepath = os.path.join(save_path, model_name + "_weights_{epoch:02d}-{accuracy:.6f}.hdf5")
    mc = ModelCheckpoint(filepath, monitor=loss_name, save_best_only=False, verbose=verbose)

    last_token_id = len(word_dict) + word_from_index

    is_training = True
    if is_training is True:

        print("Training")

        GLOBAL_BATCH_SIZE = batch_size * num_gpu
        print("batch size: ", GLOBAL_BATCH_SIZE)

        dataset = load_tfrecord(slice_files, sequence_length=word_seq_len, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(GLOBAL_BATCH_SIZE * shuffle_size, reshuffle_each_iteration=True).batch(GLOBAL_BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()

        model_train_history = albert.fit(dataset,
                                         steps_per_epoch=steps_per_epoch,
                                         epochs=epochs,
                                         callbacks=[lr_scheduler, mc],
                                         verbose=1
                                         )


