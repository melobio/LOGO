
1. Data preparation
-Download the data from http://deepsea.princeton.edu/media/code/deepsea.v0.94c.tar.gz and unzip it to /data/deepsea/

2. Convert to kmer sequence file, kmer is 3, 4, 5, 6 respectively
sh ./01_run_deepsea_tfrecord_utils.sh

3. Generate tfrecord file
sh ./01_run_deepsea_tfrecord_utils.sh

4. Carry out LOGO_Chrom_919 training and testing
sh ./02_run_deepsea_classification_train.sh

5. Weight for reproducting the paper result
deepsea_5_gram_2_layer_8_heads_256_dim_990_weights_99-0.982516-0.983271.hdf5