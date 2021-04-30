

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