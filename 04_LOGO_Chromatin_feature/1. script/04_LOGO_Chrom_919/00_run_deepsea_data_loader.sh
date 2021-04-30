#!/bin/bash

python 00_deepsea_data_loader.py \
  --data /data/deepsea/ \
  --output /data/deepsea/ \
  --ngram 6 \
  --stride 1 \
  --slice 200000 \
  --pool-size 24
