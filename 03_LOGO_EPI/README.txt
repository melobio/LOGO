
1. Data download.
- Download B cells (tB), monocytes (Mon), foetal thymus (FoeT), total CD4+ T cells (tCD4), from https://github.com/liwenran/DeepTACT
- naive CD4+ T cells (nCD4), and total CD8+ T cells (tCD8), which derived from https://osf.io/u8tzp/ (Javierre et al. 2016).
- From https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.25_GRCh37.p13/ Download GCF_000001405.25_GRCh37.p13_genomic.fna.gz,
  And unzip, for example, unzip to /data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna
- Download GCF_000001405.39_GRCh38.p13_genomic.gff from https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_genomic.gff.gz
  And unzip it to /data/hg38/GCF_000001405.39_GRCh38.p13_genomic.gff


2. Data processing and sequence generation. The generated data is about 30G.
- python 00_DataPrepare.py tB P-E
- python 00_DataPrepare.py Mon P-E
- python 00_DataPrepare.py Foet P-E
- python 00_DataPrepare.py tCD4 P-E
- python 00_DataPrepare.py nCD4 P-E
- python 00_DataPrepare.py tCD8 P-E

3. Data processing and sequence generation with knowledge. The generated data is about 30G.
- python 01_DataPrepare_Knowledge.py tB P-E
- python 01_DataPrepare_Knowledge.py Mon P-E
- python 01_DataPrepare_Knowledge.py Foet P-E
- python 01_DataPrepare_Knowledge.py tCD4 P-E
- python 01_DataPrepare_Knowledge.py nCD4 P-E
- python 01_DataPrepare_Knowledge.py tCD8 P-E

4. Generate kmer training sequence
- python 02_DataPrepare_Ngram.py

5. Generate kmer training sequence, including remarks
- python 03_DataPrepare_Ngram_Knowledge.py

6. Training and predicting EPI
- python 04_LOGO_EPI_train_conv1d_concat_atcg.py

7. Training and predicting EPI with knowledge
- python 05_LOGO_EPI_train_conv1d_concat_atcg_gene_type.py


7. Predicting EPI results
- 06_LOGO_EPI_Prediction_Result.xlsx