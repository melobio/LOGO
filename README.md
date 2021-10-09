# Languse of Genome (LOGO)

This repository contains code and pre-trained weights for ALBERT genome language models from MGI, ALBERT genome language models were introduced in our paper, [Integrating convolution and self-attention improves language model of human genome for interpreting non-coding regions at base-resolution](https://www.biorxiv.org/content/10.1101/2021.09.06.459087v1)



## Usage

### Requirement

```
## System
Ubuntu 18.04
gcc 7.5.0

## Conda environment
cudatoolkit               10.0.130                      0    defaults
cudnn                     7.6.5                cuda10.0_0    defaults
...
keras                     2.3.1                         0    defaults
keras-applications        1.0.8                      py_1    defaults
keras-base                2.3.1                    py36_0    defaults
keras-preprocessing       1.1.2              pyhd3eb1b0_0    defaults
pandas                    1.1.5            py36ha9443f7_0    defaults
python                    3.6.9                h265db76_0    defaults
...
tensorflow                2.0.0           gpu_py36h6b29c10_0    defaults
tensorflow-base           2.0.0           gpu_py36h0ec5d1f_0    defaults
tensorflow-estimator      2.0.0              pyh2649769_0    defaults
tensorflow-gpu            2.0.0                h0d30ee6_0    defaults

```



### Installation

As a prerequisite, you must have Tensorfolw-gpu 2.0.0 installed to use this repository.

You can use this three-liner for installation:

```shell
conda create --name logo python==3.6.9 tensorflow-gpu==2.0 keras==2.3.1 numpy pandas tqdm scipy scikit-learn matplotlib jupyter notebook nb_conda
source activate logo
pip install biopython==1.68
```



## Pre-training model

Check out the file “01_Pre-training_Model/README.txt”

```shell
1. Download fasta file
From https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.25_GRCh37.p13/ Download GCF_000001405.25_GRCh37.p13_genomic.fna.gz. And then unzip to ./data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna

2. To generate ref sequence, about 15G of space is required:
python 00_generate_refseq_sequence.py \
  --data ../data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna \
  --output ../data/hg19/train_5_gram \
  --chunk-size 10000 \
  --seq-size 1000 \
  --seq-stride 100 \
  --ngram 5 \
  --stride 1 \
  --slice-size 100000 \
  --hg-name hg19 \
  --pool-size 32

3. To generate tfrecord, about 237G of space is required (different kmer requires slightly different storage space, kmer=3, 4, 5, 6):
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
CUDA_VISIBLE_DEVICES=0,1,2,3 python 02_train_gene_transformer_lm_hg_bert4keras_tfrecord.py \
  --save ./data \
  --train-data /data/hg19/train_5_gram_tfrecord \
  --seq-len 1000 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 256 \
  --ngram 5 \
  --stride 5 \
  --model-name genebert_5_gram_4_layer_8_heads_256_dim \
  --steps-per-epoch 4000 \
  --shuffle-size 4000
```



## Promoter Prediction

Check out the file “02_LOGO_Promoter/README.txt”

```shell
1. Data preparation
- Download GCF_000001405.39_GRCh38.p13_genomic.gff from https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_genomic.gff.gz
  And unzip it to /data/hg38/GCF_000001405.39_GRCh38.p13_genomic.gff

- Download the file epdnew from https://epd.epfl.ch/EPD_download.php, which are BOTH, TATA_BOX, NO_TATA_BOX, and generate tfrecord,
  python 00_EPDnew_data_prepare.py

2. Make Promoter predictions
- python 01_PromID_trainer.py

3. Make Promoter + knowledge prediction
- python 02_PromID_trainer_knowledge.py

4. Promoter prediction results
- 03_LOGO_Promoter_Prediction_Result.xlsx
```



## Promoter and Enhancer Interactions Prediction

Check out the file “03_LOGO_EPI/README.txt”



## Chromatin Feature

Check out the file “04_LOGO_Chromatin_feature/README.txt”

1. script

* 04_LOGO_Chrom_919: The program code of LOGO-919, which can be used to reproduce the results of Fig 3, see its corresponding readme file for details
* 04_LOGO_Chrom_predict: The prediction program of LOGO-919/2002/3357, used to reproduce the results  in the Fig 4A and Table 1

2. result

* Demo data can be used to test the validity of the program 



## Variant Prioritization

Check out the file “05_LOGO_variant_prioritization/README.txt”



## Pre-training model weights

- 99_PreTrain_Model_Weight

