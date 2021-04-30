import os, sys
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

"""
00_DataPrepare.py
"""

sys.path.append("../")
from bgi.bert4keras.models import build_transformer_model
from bgi.common.callbacks import LRSchedulerPerStep
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number

###################### Input #######################
if len(sys.argv) < 3:
    print('[USAGE] python 00_DataPrepare.py cell interaction_type')  # py2改为py3
    print('For example, python 00_DataPrepare.py Mon P-E')
    sys.exit()
CELL = sys.argv[1]
TYPE = sys.argv[2]
TASK = sys.argv[3]

if TYPE == 'P-P':
    RESIZED_LEN = 1000  # promoter
elif TYPE == 'P-E':
    RESIZED_LEN = 2000  # enhancer
else:
    print('[USAGE] python 00_DataPrepare.py cell interaction_type')
    print('For example, python 00_DataPrepare.py Mon P-E')
    sys.exit()


include_types = ['enhancer',
                 'promoter',
                 'pseudogene',
                 'insulator',
                 'conserved_region',
                 'protein_binding_site',
                 'DNAseI_hypersensitive_site',
                 'nucleotide_cleavage_site',
                 'silencer',
                 'gene',
                 'exon',
                 'CDS',
                 # 'guide_RNA',
                 # 'rRNA',
                 # 'regulatory_region',
                 # 'snRNA',
                 # 'RNase_P_RNA',

                 # 'insulator',
                 # 'miRNA',
                 # 'microsatellite',
                 # 'vault_RNA',
                 # 'mRNA',
                 # 'tRNA',
                 # 'minisatellite',
                 # 'snoRNA',
                 # 'locus_control_region',
                 # 'CAGE_cluster',
                 # 'RNase_MRP_RNA',
                 # 'transcript',
                 # 'TATA_box',
                 # 'telomerase_RNA',
                 # 'transcriptional_cis_regulatory_region',
                 # 'antisense_RNA',
                 # 'lnc_RNA',
                 # 'Y_RNA',
                 # 'imprinting_control_region',
                 # 'enhancer_blocking_element',
                 # 'nucleotide_motif',
                 # 'primary_transcript'
                 ]

def preccess_data(seq_data: np.ndarray,
                  actg_value,
                  stride,
                  seq_max_len,
                  n_gram,
                  n_gram_value,
                  num_word_dict,
                  ):
    """
    分批处理数据，生成npz文件
    :param seq_data:
    :param actg_value:
    :param step:
    :param n_gram:
    :param n_gram_value:
    :param num_word_dict:
    :return:
    """

    # AGCT转换为1，2，3，4
    actg = np.matmul(seq_data, actg_value)
    gene = []
    for kk in range(0, len(actg), stride):
        if kk >= seq_max_len:
            break

        actg_temp_value = 0
        if kk + n_gram <= len(actg):
            actg_temp_value = np.dot(actg[kk:kk + n_gram], n_gram_value)
            actg_temp_value = int(actg_temp_value)
        else:
            for gg in range(kk, len(actg)):
                actg_temp_value += actg[gg] * (10 ** (n_gram - gg % n_gram - 1))

            # print("10 ** (kk % n_gram): ", 10 ** (kk % n_gram))
            actg_temp_value = actg_temp_value * (10 ** (kk % n_gram))

        gene.append(num_word_dict.get(actg_temp_value, 0))
    # print("gene: ", len(gene), seq_max_len, len(actg))
    return np.array(gene)


def load_npz_data(data_path, prefix='', ngram=3, reshape=True, NUM_SEQ=4, masked=True, gene_type_dict: dict = {}):
    """
    导入npz数据
    :param file_name:
    :param ngram:
    :param only_one_slice:
    :param ngram_index:
    :return:
    """
    x_data_all = []
    x_annotation_all = []
    y_data_all = []
    file_names = []

    actg_value = np.array([1, 2, 3, 4])
    n_gram_value = np.ones(ngram)
    for ii in range(ngram):
        n_gram_value[ii] = int(n_gram_value[ii] * (10 ** (ngram - ii - 1)))
    print("n_gram_value: ", n_gram_value)

    num_word_dict = get_word_dict_for_n_gram_number(n_gram=ngram)

    files = os.listdir(data_path)
    for file_name in files:
        print(file_name)
        if str(file_name).startswith(prefix) and str(file_name).endswith('knowledge.npz'):
            file_names.append(os.path.join(data_path, file_name))

            loaded = np.load(os.path.join(data_path, file_name), allow_pickle=True)
            x_data = loaded['sequence']
            annotation = loaded['annotation']
            y_data = loaded['label']

            print("x_data: ", x_data.shape)
            print("annotation: ", annotation.shape)
            print("y_data: ", y_data.shape)

            x_data_all.append(x_data)
            x_annotation_all.append(annotation)
            y_data_all.append(y_data)
            # break

    x_data_all = np.vstack(x_data_all)

    print("annotation_all: ", x_annotation_all)
    x_annotation_all = np.array(x_annotation_all)
    x_annotation_all = np.reshape(x_annotation_all, (x_annotation_all.shape[-1]))
    print("x_data_all: ", x_data_all.shape)
    print("annotation_all: ", x_annotation_all.shape)
    x_gene_seq_all = []
    x_gene_annotation_all = []
    if reshape is True:
        x_data_all = np.reshape(x_data_all, (x_data_all.shape[0], x_data_all.shape[1] // NUM_SEQ, NUM_SEQ))
        seq_max_len = x_data_all.shape[1] // ngram // ngram * ngram * ngram

        for ii in range(len(x_data_all)):
            seq_data = x_data_all[ii]
            gene_seq = preccess_data(seq_data,
                                     actg_value=actg_value,
                                     stride=1,
                                     seq_max_len=seq_max_len,
                                     n_gram=ngram,
                                     n_gram_value=n_gram_value,
                                     num_word_dict=num_word_dict,
                                     )
            x_gene_seq_all.append(gene_seq)


            # 序列注释信息
            seq_size = len(gene_seq)
            annotation = x_annotation_all[ii]

            # print("annotation: ", annotation)

            annotation = np.array(annotation)
            annotation_len = len(annotation)
            annotation_position = np.zeros((len(gene_type_dict.keys()) + 2, seq_size), dtype=np.int)
            if annotation_len > 0:
                for jj in range(annotation_len):
                    gene_type = annotation[jj][0]
                    gene_type = gene_type_dict.get(gene_type, 0)
                    start = int(annotation[jj][1])
                    end = int(annotation[jj][2])
                    annotation_position[gene_type, start:min(seq_size, end)] = 1

            x_gene_annotation_all.append(annotation_position)
            if ii > 0 and ii % 10000 == 0:
                print(ii)
        x_data_all = np.array(x_gene_seq_all)
        x_annotation_all = np.array(x_gene_annotation_all)

    print("x_annotation_all: ", x_annotation_all[0])
    y_data_all = np.hstack(y_data_all)

    return x_data_all, x_annotation_all, y_data_all


def main():
    # CELL = 'tB'
    # TYPE = 'P-E'

    ngram = 6

    gene_type_dict = {}
    index = 1  # 0 means unknown
    for gene_type in include_types:
        gene_type_dict[gene_type] = index
        index += 1

    # task = 'test'
    ## load data: sequence
    if TASK == 'train':
        seq_data_path = CELL + '/' + TYPE + '/{}_gram/'.format(str(ngram))
        if os.path.exists(seq_data_path) is False:
            os.makedirs(seq_data_path)

        data_path = CELL + '/' + TYPE + '/'
        region1_seq, annotation, label = load_npz_data(data_path, ngram=ngram, prefix='enhancer_Seq', gene_type_dict=gene_type_dict)
        print("region1_seq: ", region1_seq.shape)
        np.savez(os.path.join(seq_data_path, 'enhancer_Seq_{}_gram_knowledge.npz'.format(ngram)), x=region1_seq, y=label, annotation=annotation)

        region1_seq, annotation, label = load_npz_data(data_path, ngram=ngram, prefix='promoter_Seq', gene_type_dict=gene_type_dict)
        np.savez(os.path.join(seq_data_path, 'promoter_Seq_{}_gram_knowledge.npz'.format(ngram)), x=region1_seq, y=label, annotation=annotation)
    else:
        seq_data_path = CELL + '/' + TYPE + '/test/{}_gram/'.format(str(ngram))
        if os.path.exists(seq_data_path) is False:
            os.makedirs(seq_data_path)

        data_path = CELL + '/' + TYPE + '/test/'
        region1_seq, annotation, label = load_npz_data(data_path, ngram=ngram, prefix='enhancer_Seq', gene_type_dict=gene_type_dict)
        print("region1_seq: ", region1_seq.shape)
        np.savez(os.path.join(seq_data_path, 'enhancer_Seq_{}_gram_knowledge.npz'.format(ngram)), x=region1_seq, y=label, annotation=annotation)

        region1_seq, annotation, label = load_npz_data(data_path, ngram=ngram, prefix='promoter_Seq', gene_type_dict=gene_type_dict)
        np.savez(os.path.join(seq_data_path, 'promoter_Seq_{}_gram_knowledge.npz'.format(ngram)), x=region1_seq, y=label, annotation=annotation)






"""RUN"""
main()
