
1. Data download, see Data_URL.txt, Save to /data/CADD/

2. Generate vcf sequence

- python 00_cadd_create_vcf_sequence_data_multi_thread.py

3. Generate tfrecord, about 30 G

- python 01_cadd_create_tfrecord.py

4. Perform training and prediction
- python 02_cadd_classification_transformer_tfrecord.py

