
1. From https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.25_GRCh37.p13/ Download GCF_000001405.25_GRCh37.p13_genomic.fna.gz,
And unzip, for example, unzip to /data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna

2. To generate ref sequence, , about 100G of space is required:
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

3. To generate tfrecord, about 170G of space is required (different kmer requires slightly different storage space, kmer=3, 4, 5, 6):
python 01_generate_DNA_refseq_tfrecord.py \
  --output /data/hg19/train_5_gram \
  --output /data/hg19/train_5_gram_tfrecord \
  --chunk-size 10000 \
  --seq-size 1000 \
  --seq-stride 100 \
  --ngram 5 \
  --stride 1 \
  --slice-size 100000 \
  --hg-name hg19 \
  --pool-size 32

4. Perform DNA sequence pre-training, respectively (kmer=3,4,5,6, perform training):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup 03_train_gene_transformer_lm_hg_bert4keras_tfrecord.py \
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