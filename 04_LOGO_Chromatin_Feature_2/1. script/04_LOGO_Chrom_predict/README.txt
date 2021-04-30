Chrom-919/2002/3357 Chromatin Features Prediction Explanation


- 1. inputfile, vcf file, delimiter is \t
chr1 60351 rs62637817 A G
chr1 135173 rs575342270 G A

- 2. backgroundfile, background distribution file, in this directory "2.background", 1million snp is used as the background mutation, calculated with the help of this software, because the file is too large, a link will be provided after the subsequent upload is successful, the size is as follows:

cd /alldata/Nzhang_data/project/T2D/2.background
23G 1.2002mark_5gram
38G 2.3357mark_5gram
21G 5.919mark_5gram
41G 6.3540mark_5gram

- 3. reffasta, hg19 reference genome, "LOGO\Genomics"

- 4. weight-path, the weight file of Chrom-919/2002/3357/3540 model, ending with .hdf5

- 5. For other parameters, please refer to the parameter help file of GeneBert_predict_vcf_slice_e8.py

## demo
source activate tf20_hhp
CUDA_VISIBLE_DEVICES=${2} python GeneBert_predict_vcf_slice_e8.py \
--inputfile ${1} \
--backgroundfile /../2.background/2.3357mark_5gram/ \
--reffasta Genomics/male.hg19.fasta \
--maxshift 0 \
--weight-path ./genebert_5_gram_2_layer_8_heads_256_dim_wanrenexp_tfrecord_5_1_[baseon27]_weights_119-0.9748-0.9765.hdf5 \
--seq-len 2000 \
--model-dim 256 \
--transformer-depth 2 \
--num-heads 8 \
--batch-size 128 \
--num-classes 3357 \
--shuffle-size 4000 \
--pool-size 20 \
--slice-size 5000 \
--ngram 5 \
--stride 1