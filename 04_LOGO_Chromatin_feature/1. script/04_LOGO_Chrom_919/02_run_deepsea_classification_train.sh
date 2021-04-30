#!/bin/bash


CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python deep_sea_train_classification_tfrecord.py \
  --save ../../data \
  --weight-path ../../data/genebert_5_gram_2_layer_8_heads_256_dim_bert_generator_weights_100-0.885734.hdf5 \
  --train-data /data/deepsea/deepsea_train/train_5_gram_classification_990 \
  --test-data /data/deepsea/deepsea_train/test_5_gram_classification_990 \
  --valid-data /data/deepsea/deepsea_train/valid_5_gram_classification_990 \
  --seq-len 990 \
  --we-size 256 \
  --model-dim 256 \
  --transformer-depth 1 \
  --num-heads 8 \
  --batch-size 512 \
  --ngram 5 \
  --stride 1 \
  --num-classes 919 \
  --model-name deepsea_5_gram_2_layer_8_heads_256_dim_990 \
  --steps-per-epoch 4000 \
  --shuffle-size 4000 \
  --use-conv \
  --use-position \
  --verbose 1 \
  --task train > deepsea_5_gram_1_layer_8_heads_256_dim_result_2.txt &
