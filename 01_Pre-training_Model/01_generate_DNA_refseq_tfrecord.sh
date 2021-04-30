#!/bin/bash

python 01_generate_DNA_refseq_tfrecord.py \
  --data /data/hg19/train_5_gram \
  --output /data/hg19/train_5_gram_tfrecord \
  --chunk-size 10000 \
  --seq-size 1000 \
  --seq-stride 100 \
  --ngram 5 \
  --stride 1 \
  --slice-size 100000 \
  --hg-name hg19 \
  --pool-size 32

