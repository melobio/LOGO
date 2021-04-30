import os, sys
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
import re


sys.path.append("../")
from bgi.common.genebank_utils import get_refseq_gff, get_gene_feature_array

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

chr_dict = {"NC_000001.10": "chr1",
            "NC_000002.11": "chr2",
            "NC_000003.11": "chr3",
            "NC_000004.11": "chr4",
            "NC_000005.9": "chr5",
            "NC_000006.11": "chr6",
            "NC_000007.13": "chr7",
            "NC_000008.10": "chr8",
            "NC_000009.11": "chr9",
            "NC_000010.10": "chr10",
            "NC_000011.9": "chr11",
            "NC_000012.11": "chr12",
            "NC_000013.10": "chr13",
            "NC_000014.8": "chr14",
            "NC_000015.9": "chr15",
            "NC_000016.9": "chr16",
            "NC_000017.10": "chr17",
            "NC_000018.9": "chr18",
            "NC_000019.9": "chr19",
            "NC_000020.10": "chr20",
            "NC_000021.8": "chr21",
            "NC_000022.10": "chr22",
            "NC_000023.10": "chrX",
            "NC_000024.9": "chrY"}

def split():
    pairs = pd.read_csv(CELL + '/' + TYPE + '/pairs.csv')
    n_sample = pairs.shape[0]
    rand_index = list(range(0, n_sample))
    np.random.seed(n_sample)
    np.random.shuffle(rand_index)

    n_sample_train = n_sample - n_sample // 10  # Take 90% as train
    pairs_train = pairs.iloc[rand_index[:n_sample_train]]  # Divide the data set
    pairs_test = pairs.iloc[rand_index[n_sample_train:]]

    # imbalanced testing set
    pairs_test_pos = pairs_test[pairs_test['label'] == 1]  # Take pos and nneg in tets
    pairs_test_neg = pairs_test[pairs_test['label'] == 0]
    num_pos = pairs_test_pos.shape[0]
    num_neg = pairs_test_neg.shape[0]
    np.random.seed(num_neg)
    rand_index = list(range(0, num_neg))

    pairs_test_neg = pairs_test_neg.iloc[rand_index[:num_pos * 5]]
    pairs_test = pd.concat([pairs_test_pos, pairs_test_neg])

    # save
    pairs_train.to_csv(CELL + '/' + TYPE + '/pairs_train.csv', index=False)
    print("Writting ", CELL + '/' + TYPE + '/pairs_train.csv')
    pairs_test.to_csv(CELL + '/' + TYPE + '/pairs_test.csv', index=False)
    print("Writting ", CELL + '/' + TYPE + '/pairs_test.csv')


def resize_location(original_location, resize_len):
    original_len = int(original_location[1]) - int(original_location[0])
    len_diff = abs(resize_len - original_len)
    rand_int = np.random.randint(0, len_diff + 1)
    if resize_len < original_len: rand_int = - rand_int
    resize_start = int(original_location[0]) - rand_int
    resize_end = resize_start + resize_len
    return (str(resize_start), str(resize_end))


def augment():
    RESAMPLE_TIME = 20
    PROMOTER_LEN = 1000
    fout = open(CELL + '/' + TYPE + '/pairs_train_augment.csv', 'w')
    file = open(CELL + '/' + TYPE + '/pairs_train.csv')
    for line in file:
        line = line.strip().split(',')
        if line[-1] != '1':
            fout.write(','.join(line) + '\n')
            continue
        for j in range(0, RESAMPLE_TIME):
            # Reshape the original sequence less than 2kbp into 2kbp length
            original_location = (line[1], line[2])
            resized_location = resize_location(original_location, RESIZED_LEN)                       # enhancer 2kbp
            fout.write(','.join([line[0], resized_location[0], resized_location[1], line[3]]) + ',')

            # Reshape the original sequence less than 2kbp into 2kbp length
            original_location = (line[5], line[6])
            resized_location = resize_location(original_location, PROMOTER_LEN)                      # promoter 1kbp
            fout.write(','.join([line[0], resized_location[0], resized_location[1], line[3]]) + ',1,\n')
    print("Finished:", CELL + '/' + TYPE + '/pairs_train_augment.csv')
    file.close()
    fout.close()


def one_hot(sequence_dict, chrom, start, end, chr_convert_dict:dict={}):
    seq_dict = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0],
                'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
                'a': [1, 0, 0, 0], 'g': [0, 1, 0, 0],
                'c': [0, 0, 1, 0], 't': [0, 0, 0, 1]}
    temp = []



    seq = str(sequence_dict[chr_convert_dict[chrom]].seq[start:end])
    seq = seq.upper()
    # print("seq")
    for c in seq:
        temp.extend(seq_dict.get(c, [0, 0, 0, 0]))
    temp = np.array(temp)
    # print("temp: ", temp.shape)
    return temp


def encoding(sequence_dict, filename, PROMOTER_LEN = 1000, NUM_SEQ = 4, chr_gff_dict=None, task:str=None):
    file = open(CELL + '/' + TYPE + '/' + filename)
    file.readline()
    seqs_1 = []
    seqs_2 = []
    seqs_1_annotation = []
    seqs_2_annotation = []
    label = []

    chr_convert_dict = {}
    for k, v in chr_dict.items():
        chr_convert_dict[v] = k

    # Extract sequence one by one
    ii = 0
    for line in tqdm(file):
        if ii == 0:
            ii += 1
            continue

        line = line.strip().split(',')



        seq_1 = one_hot(sequence_dict, line[0], int(line[1]), int(line[2]), chr_convert_dict)
        seq_2 = one_hot(sequence_dict, line[4], int(line[5]), int(line[6]), chr_convert_dict)

        seq_1_annotation = get_gene_feature_array(chr_gff_dict, chr_convert_dict.get(line[0], ''), int(line[1]), int(line[2]))
        seq_2_annotation = get_gene_feature_array(chr_gff_dict, chr_convert_dict.get(line[4], ''), int(line[5]), int(line[6]))
        # print("Annotation: ", seq_1_annotation)

        if len(seq_1) != RESIZED_LEN * NUM_SEQ or len(seq_2) != PROMOTER_LEN * NUM_SEQ:
            print(len(seq_1), len(seq_2))
            continue

        if len(line[-1]) == 0:
            if len(line[-2]) > 0:
                label.append(int(line[-2]))  # The last one is label
            else:
                print(line)
                continue
        else:
            label.append(int(line[-1]))  # The last one is label

        seqs_1.append(seq_1)  # Extract the first sequence (such as P)
        seqs_2.append(seq_2)  # Extract the second sequence (such as E)
        seqs_1_annotation.append(seq_1_annotation)  # Extract the first sequence (such as P)
        seqs_2_annotation.append(seq_2_annotation)  # Extract the second sequence (such as E)

        # label.append(int(line[-1]))  # The last one is label
        ii += 1

        if len(seqs_1) % 50000 == 0:
            if TYPE == 'P-P':
                print("promoter1_Seq, promoter2_Seq, label shape : ",
                      np.array(seqs_1).shape,
                      np.array(seqs_2).shape,
                      np.array(label).shape)
                np.savez(CELL + '/' + TYPE + '/promoter1_Seq_{}.npz'.format(str(ii)),
                         label=np.array(label),
                         sequence=np.array(seqs_1),
                         annotation=np.array(seqs_1_annotation))
                np.savez(CELL + '/' + TYPE + '/promoter2_Seq_{}.npz'.format(str(ii)),
                         label=np.array(label),
                         sequence=np.array(seqs_2),
                         annotation=np.array(seqs_2_annotation))
            else:
                print("enhancer_Seq, promoter_Seq, label shape : ",
                      np.array(seqs_1).shape,
                      np.array(seqs_2).shape,
                      np.array(label).shape)
                np.savez(CELL + '/' + TYPE + '/enhancer_Seq_{}.npz'.format(str(ii)),
                         label=np.array(label),
                         sequence=np.array(seqs_1),
                         annotation=np.array(seqs_1_annotation))
                np.savez(CELL + '/' + TYPE + '/promoter_Seq_{}.npz'.format(str(ii)),
                         label=np.array(label),
                         sequence=np.array(seqs_2),
                         annotation=np.array(seqs_2_annotation))

            seqs_1 = []
            seqs_2 = []
            seqs_1_annotation = []
            seqs_2_annotation = []
            label = []

    if TYPE == 'P-P':
        print("promoter1_Seq, promoter2_Seq, label shape : ",
              np.array(seqs_1).shape,
              np.array(seqs_2).shape,
              np.array(label).shape)
        np.savez(CELL + '/' + TYPE + '/promoter1_Seq_{}.npz'.format(str(ii)),
                 label=np.array(label),
                 sequence=np.array(seqs_1),
                 annotation=np.array(seqs_1_annotation))
        np.savez(CELL + '/' + TYPE + '/promoter2_Seq_{}.npz'.format(str(ii)),
                 label=np.array(label),
                 sequence=np.array(seqs_2),
                 annotation=np.array(seqs_2_annotation))
    else:
        print("enhancer_Seq, promoter_Seq, label shape : ",
              np.array(seqs_1).shape,
              np.array(seqs_2).shape,
              np.array(label).shape)
        np.savez(CELL + '/' + TYPE + '/enhancer_Seq_{}.npz'.format(str(ii)),
                 label=np.array(label),
                 sequence=np.array(seqs_1),
                 annotation=np.array(seqs_1_annotation))
        np.savez(CELL + '/' + TYPE + '/promoter_Seq_{}.npz'.format(str(ii)),
                 label=np.array(label),
                 sequence=np.array(seqs_2),
                 annotation=np.array(seqs_2_annotation))


def encoding_test(sequence_dict,
                  filename,
                  PROMOTER_LEN = 1000,
                  NUM_SEQ = 4,
                  task:str=None,
                  chr_gff_dict: dict = {}):
    file = open(CELL + '/' + TYPE + '/' + filename)
    file.readline()
    seqs_1 = []
    seqs_2 = []
    seqs_1_annotation = []
    seqs_2_annotation = []
    label = []

    chr_convert_dict = {}
    for k, v in chr_dict.items():
        chr_convert_dict[v] = k

    if os.path.exists(CELL + '/' + TYPE + '/test/') is False:
        os.makedirs(CELL + '/' + TYPE + '/test/')

    # Extract sequence one by one
    ii = 0
    for line in tqdm(file):
        if ii == 0:
            ii += 1
            continue

        line = line.strip().split(',')


        convert_chr = chr_convert_dict.get(line[0], '')
        seq_1 = one_hot(sequence_dict, line[0], int(line[1]), int(line[2]), chr_convert_dict)
        seq_2 = one_hot(sequence_dict, line[4], int(line[5]), int(line[6]), chr_convert_dict)

        seq_1_annotation = get_gene_feature_array(chr_gff_dict, chr_convert_dict.get(line[0], ''), int(line[1]), int(line[2]))
        seq_2_annotation = get_gene_feature_array(chr_gff_dict, chr_convert_dict.get(line[4], ''), int(line[5]), int(line[6]))
        #print("Annotation1: ", seq_1_annotation)
        #print("Annotation2: ", type(seq_2_annotation), seq_2_annotation)


        if len(seq_1) != RESIZED_LEN * NUM_SEQ or len(seq_2) != PROMOTER_LEN * NUM_SEQ:
            print(len(seq_1), len(seq_2))
            continue

        if len(line[-1]) == 0:
            if len(line[-2]) > 0:
                label.append(int(line[-2]))  # The last one is label
            else:
                print(line)
                continue
        else:
            label.append(int(line[-1])) # The last one is label

        seqs_1.append(seq_1)  # Extract the first sequence (such as P)
        seqs_2.append(seq_2)  # Extract the second sequence (such as E)
        seqs_1_annotation.append(seq_1_annotation)  # Extract the first sequence (such as P)
        seqs_2_annotation.append(seq_2_annotation)  # Extract the second sequence (such as E)


        # label.append(int(line[-1]))  # The last one is label
        ii += 1

        if len(seqs_1) % 50000 == 0:
            if TYPE == 'P-P':
                print("promoter1_Seq, promoter2_Seq, label shape : ",
                      np.array(seqs_1).shape,
                      np.array(seqs_2).shape,
                      np.array(label).shape)
                np.savez(CELL + '/' + TYPE + '/test/promoter1_Seq_{}_knowledge.npz'.format(str(ii)),
                         label=np.array(label),
                         sequence=np.array(seqs_1),
                         annotation=np.array(seqs_1_annotation)
                         )
                np.savez(CELL + '/' + TYPE + '/test/promoter2_Seq_{}_knowledge.npz'.format(str(ii)),
                         label=np.array(label),
                         sequence=np.array(seqs_2),
                         annotation=np.array(seqs_2_annotation)
                         )
            else:
                print("enhancer_Seq, promoter_Seq, label shape : ",
                      np.array(seqs_1).shape,
                      np.array(seqs_2).shape,
                      np.array(label).shape)
                np.savez(CELL + '/' + TYPE + '/test/enhancer_Seq_{}_knowledge.npz'.format(str(ii)),
                         label=np.array(label),
                         sequence=np.array(seqs_1),
                         annotation=np.array(seqs_1_annotation))
                np.savez(CELL + '/' + TYPE + '/test/promoter_Seq_{}_knowledge.npz'.format(str(ii)),
                         label=np.array(label),
                         sequence=np.array(seqs_2),
                         annotation=np.array(seqs_2_annotation))

            seqs_1 = []
            seqs_2 = []
            seqs_1_annotation = []
            seqs_2_annotation = []
            label = []

    if TYPE == 'P-P':
        print("promoter1_Seq, promoter2_Seq, label shape : ",
              np.array(seqs_1).shape,
              np.array(seqs_2).shape,
              np.array(label).shape)
        np.savez(CELL + '/' + TYPE + '/test/promoter1_Seq_{}_knowledge.npz'.format(str(ii)),
                 label=np.array(label),
                 sequence=np.array(seqs_1),
                 annotation=np.array(seqs_1_annotation))
        np.savez(CELL + '/' + TYPE + '/test/promoter2_Seq_{}_knowledge.npz'.format(str(ii)),
                 label=np.array(label),
                 sequence=np.array(seqs_2),
                 annotation=np.array(seqs_2_annotation))
    else:
        print("enhancer_Seq, promoter_Seq, label shape : ",
              np.array(seqs_1).shape,
              np.array(seqs_2).shape,
              np.array(label).shape)
        np.savez(CELL + '/' + TYPE + '/test/enhancer_Seq_{}_knowledge.npz'.format(str(ii)),
                 label=np.array(label),
                 sequence=np.array(seqs_1),
                 annotation=np.array(seqs_1_annotation)
                 )
        np.savez(CELL + '/' + TYPE + '/test/promoter_Seq_{}_knowledge.npz'.format(str(ii)),
                 label=np.array(label),
                 sequence=np.array(seqs_2),
                 annotation=np.array(seqs_2_annotation)
                 )


def main():

    gff_file = '/data/hg19/GCF_000001405.25_GRCh37.p13_genomic.gff'

    # Annotate the signature file
    chr_gff_dict = get_refseq_gff(gff_file, include_types)
    chr_convert_dict = {}
    for k, v in chr_dict.items():
        chr_convert_dict[v] = k


    if TASK == 'train':
        """Split for training and testing data"""
        print("Split for training and testing data")
        split()
        """Augment training data"""
        print("Augment training data")
        augment()
        """One-hot encoding"""
        print("One-hot encoding")
        reffasta = '/data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna'

        sequence_dict = SeqIO.to_dict(SeqIO.parse(open(reffasta), 'fasta'))
        # sequence_dict = SeqIO.to_dict(SeqIO.parse(open('E:/myP/ExPecto/resources/hg19.fa'), 'fasta'))
        encoding(sequence_dict, 'pairs_train_augment.csv', chr_gff_dict=chr_gff_dict)
        print("Finished!")
    else:
        print("One-hot encoding")
        reffasta = '/data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna'
        sequence_dict = SeqIO.to_dict(SeqIO.parse(open(reffasta), 'fasta'))
        encoding_test(sequence_dict, 'pairs_test.csv', chr_gff_dict=chr_gff_dict)
        print("Finished!")


"""RUN"""
main()
