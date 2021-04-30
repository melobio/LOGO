#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup 02_train_gene_transformer_lm_hg_bert4keras_tfrecord.py \
  --save ./data \
  --train-data /data/hg19/train_5_gram_tfrecord \
  --seq-len 1000 \
  --model-dim 256 \
  --transformer-depth 4 \
  --num-heads 8 \
  --batch-size 256 \
  --ngram 5 \
  --stride 1 \
  --model-name genebert_5_gram_4_layer_8_heads_256_dim \
  --steps-per-epoch 4000 \
  --shuffle-size 4000 > genebert_5_gram_4_layer_8_heads_256_dim_result.txt &

