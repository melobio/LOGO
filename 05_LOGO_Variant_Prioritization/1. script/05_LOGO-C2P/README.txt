
Use LOGO C2P predict snp

xxx.html is the html presentation of the implementation process, and xxx.ipynb is the notebook

Take "GWAS_C2P" as an example

* main pipeline code: "GWAS_1408.ipynb"

	1. get LOGO-919/2002 chromatin feature

	2. get evolution feature by "run_evo.sh", need download our docker images

	3. train xgboost model

* Output file: "1000G_GWAS_1408.vcf_output"
	1000G_GWAS_1408.vcf.evoall.wholerow
	1000G_GWAS_1408.vcf_128bs_5gram_919feature.out.logfoldchange.csv
	1000G_GWAS_1408.vcf_128bs_5gram_919feature.out.diff.csv

* wegith come from HGMD-LOGO-C2P model: "xgboost_model_weight"
	1000G_HGMD_posstrand_8softwares_5_test_shuffle8_XGboost_919mark_Trible.model
	1000G_HGMD_posstrand_8softwares_5_test_shuffle8_XGboost_2002mark_Trible.model
	1000G_HGMD_posstrand_8softwares_5_test_shuffle8_XGboost_3357mark_Trible.model