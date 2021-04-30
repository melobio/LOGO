#!/bin/bash

python 00_generate_refseq_sequence.py \
  --data /data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna \
  --output /data/hg19/train_5_gram \
  --chunk-size 10000 \
  --seq-size 1000 \
  --seq-stride 100 \
  --ngram 5 \
  --stride 1 \
  --slice-size 100000 \
  --hg-name hg19 \
  --pool-size 32

