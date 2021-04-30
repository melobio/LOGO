#!/bin/bash

python deepsea_tfrecord_utils.py \
  --data /data/deepsea/deepsea_train/train_5_gram_990 \
  --output /data/deepsea/deepsea_train/train_5_gram_classification_tfrecord_990 \
  --ngram 5 \
  --stride 1 \
  --slice 200000 \
  --pool-size 24 \
  --task classification



python deepsea_tfrecord_utils.py \
  --data /data/deepsea/deepsea_train/test_5_gram_990 \
  --output /data/deepsea/deepsea_train/test_5_gram_classification_tfrecord_990 \
  --ngram 5 \
  --stride 1 \
  --slice 200000 \
  --pool-size 24 \
  --task classification


python deepsea_tfrecord_utils.py \
  --data /data/deepsea//deepsea_train/valid_5_gram_990 \
  --output /data/deepsea/deepsea_train/valid_5_gram_classification_tfrecord_990 \
  --ngram 5 \
  --stride 1 \
  --slice 200000 \
  --pool-size 24 \
  --task classification
