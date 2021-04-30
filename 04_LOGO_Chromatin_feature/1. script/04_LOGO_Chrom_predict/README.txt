Chrom-919/2002/3357染色质特征预测说明

* 用法：
	* 给定一个vcf文件，预测vcf中每个变异的evalue

* 输入
	* 1. inputfile, vcf file, 分隔符为\t
	
chr1    60351   rs62637817      A       G
chr1    135173  rs575342270     G       A

	* 2. backgroundfile, 背景分布文件，在本目录"2.background"，将1million snp作为背景变异，借助本软件计算后得到，由于文件太大，后续上传成功后将提供链接，大小如下所示：

cd /alldata/Nzhang_data/project/T2D/2.background
23G     1.2002mark_5gram
38G     2.3357mark_5gram
21G     5.919mark_5gram
41G     6.3540mark_5gram

	* 3. reffasta，hg19参考基因组，"LOGO\Genomics"
	
	* 4. weight-path，Chrom-919/2002/3357/3540模型的权重文件，以.hdf5结尾
	
	* 5. 其他参数，见GeneBert_predict_vcf_slice_e8.py的参数帮助文档即可

# demo
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