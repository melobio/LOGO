# LChuang: USAGE 
# (tf20_hhp) LChuang@bio-SYS-4028GR-TR:/alldata/LChuang_data/myP/GeneBert/BGI-Gene_new/examples/vcf-predict$
# USAGE: bash run_GeneBert_919_2002_3357.sh demo.vcf 2
# input:
#   demo.vcf
#   GPU=2
# Output:
#   .out.ref.csv
#   .out.alt.csv
#   .out.logfoldchange.csv
#   .out.evalue.csv
#   .out.evalue_gmean.csv   # main output

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


#CUDA_VISIBLE_DEVICES=${2} python GeneBert_predict_vcf_slice_e8-512pos_new.py \
#--inputfile ${1} \
#--backgroundfile /alldata/Nzhang_data/project/T2D/2.background/5.919mark_5gram/ \
#--reffasta /alldata/LChuang_data/workspace/Project/GeneBERT/BGI-Gene_new/male.hg19.fasta \
#--maxshift 0 \
#--weight-path /alldata/LChuang_data/workspace/Project/GeneBERT/BGI-Gene_new/data_deepsea_mat/deepsea_5_gram_2_layer_8_heads_256_dim_weights_128-0.982105-0.983445.hdf5 \
#--seq-len 1000 \
#--model-dim 256 \
#--transformer-depth 2 \
#--num-heads 8 \
#--batch-size 128 \
#--num-classes 919 \
#--shuffle-size 4000 \
#--pool-size 36 \
#--slice-size 10000 \
#--ngram 5 \
#--stride 1


#CUDA_VISIBLE_DEVICES=${2} python GeneBert_predict_vcf_slice_e8.py \
#--inputfile ${1} \
#--backgroundfile /alldata/Nzhang_data/project/T2D/2.background/1.2002mark_5gram/ \
#--reffasta /alldata/LChuang_data/workspace/Project/GeneBERT/BGI-Gene_new/male.hg19.fasta \
#--maxshift 0 \
#--weight-path /alldata/LChuang_data/workspace/Project/GeneBERT/BGI-Gene_new/data_exp_mat_5_1/genebert_5_gram_2_layer_8_heads_256_dim_expecto_[mat]_weights_134-0.9633-0.9671.hdf5 \
#--seq-len 2000 \
#--model-dim 256 \
#--transformer-depth 2 \
#--num-heads 8 \
#--batch-size 128 \
#--num-classes 2002 \
#--shuffle-size 4000 \
#--pool-size 25 \
#--slice-size 5000 \
#--ngram 5 \
#--stride 1

## old sigleGPU 120lun
# CUDA_VISIBLE_DEVICES=${2} python GeneBert_predict_vcf_slice_e8.py \
# --inputfile ${1} \
# --backgroundfile /alldata/Nzhang_data/project/T2D/2.background/6.3540mark_5gram/ \
# --reffasta /alldata/LChuang_data/workspace/Project/GeneBERT/BGI-Gene_new/male.hg19.fasta \
# --maxshift 0 \
# --weight-path /alldata/LChuang_data/workspace/Project/GeneBERT/BGI-Gene_new/data_wanrenexp_3540_selene_ngram_5_1/genebert_5_gram_2_layer_8_heads_256_dim_wanrenexp_tfrecord_5_1_weights_120-0.9753-0.9773.hdf5 \
# --seq-len 2000 \
# --model-dim 256 \
# --transformer-depth 2 \
# --num-heads 8 \
# --batch-size 128 \
# --num-classes 3540 \
# --shuffle-size 4000 \
# --pool-size 10 \
# --slice-size 5000 \
# --ngram 5 \
# --stride 1

#CUDA_VISIBLE_DEVICES=${2} python GeneBert_predict_vcf_slice_e8.py \
#--inputfile ${1} \
#--backgroundfile /alldata/Nzhang_data/project/T2D/2.background/6.3540mark_5gram/ \
#--reffasta /alldata/LChuang_data/workspace/Project/GeneBERT/BGI-Gene_new/male.hg19.fasta \
#--maxshift 0 \
#--weight-path /alldata/LChuang_data/workspace/Project/GeneBERT/BGI-Gene_new/data_wanrenexp_3540_selene_ngram_5_1_4GPU/genebert_5_gram_2_layer_8_heads_256_dim_wanrenexp_tfrecord_5_1_Baseon_12run_weights_120-0.9764-0.9774.hdf5 \
#--seq-len 2000 \
#--model-dim 256 \
#--transformer-depth 2 \
#--num-heads 8 \
#--batch-size 128 \
#--num-classes 3540 \
#--shuffle-size 4000 \
#--pool-size 10 \
#--slice-size 5000 \
#--ngram 5 \
#--stride 1
