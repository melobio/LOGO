from pyfaidx import Fasta
import random
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../")
from bgi.common.refseq_utils import get_word_dict_for_n_gram_alphabet
from bgi.common.genebank_utils import get_refseq_gff, get_gene_feature_array

include_types = ['enhancer',
                 #'promoter',
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
# HG19
chr_dict_hg19 = {"NC_000001.10": "chr1",
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

# HG38
chr_dict_hg38 = {"NC_000001.11": "chr1",
                 "NC_000002.12": "chr2",
                 "NC_000003.12": "chr3",
                 "NC_000004.12": "chr4",
                 "NC_000005.10": "chr5",
                 "NC_000006.12": "chr6",
                 "NC_000007.14": "chr7",
                 "NC_000008.11": "chr8",
                 "NC_000009.12": "chr9",
                 "NC_000010.11": "chr10",
                 "NC_000011.10": "chr11",
                 "NC_000012.12": "chr12",
                 "NC_000013.11": "chr13",
                 "NC_000014.9": "chr14",
                 "NC_000015.10": "chr15",
                 "NC_000016.10": "chr16",
                 "NC_000017.11": "chr17",
                 "NC_000018.10": "chr18",
                 "NC_000019.10": "chr19",
                 "NC_000020.11": "chr20",
                 "NC_000021.9": "chr21",
                 "NC_000022.11": "chr22",
                 "NC_000023.11": "chrX",
                 "NC_000024.10": "chrY"}


def get_negative(sequence: str,
                 TSS_position=5000,
                 low_position=200,
                 high_position=400,
                 ngram=3
                 ):
    seq_len = len(sequence)

    while (True):
        left_position = random.randint(0, TSS_position)
        if left_position - low_position >= 0 and left_position + high_position < (TSS_position - low_position):
            start = left_position - low_position
            end = left_position + high_position + (ngram - 1)
            # print("Left ......")
            return sequence[start:end], left_position

        right_position = random.randint(TSS_position, seq_len)
        if right_position - low_position > TSS_position + high_position and right_position + high_position + (
                ngram - 1) < seq_len:
            start = right_position - low_position
            end = right_position + high_position + (ngram - 1)
            # print("Right ......")
            return sequence[start:end], left_position


def process_raw_text(sequences,
                     labels,
                     seq_size=1000,
                     ngram=3,
                     stride=1,
                     filter_txt=None,
                     skip_n: bool = False,
                     word_dict: dict = None,
                     output_path: str = './',
                     task_name: str = 'train',
                     gene_type_dict: dict = None):
    slice_index = 0
    slice_seq_data = []
    slice_label_data = []

    set_atcg = set(list('ATCG'))

    print("seq_size: ", seq_size)

    for row in range(len(sequences)):
        seq = sequences[row]
        label = labels[row]

        # print("SEQ: ", seq)
        seq = seq.upper()
        if filter_txt is not None and seq.startswith(filter_txt):
            continue

        if skip_n is True:
            if seq.find('N') > -1:
                print(seq)
                continue

        # Check if it’s not ‘ATCG’
        set_seq = set(list(seq))
        is_atcg = True
        for atcg in set_seq:
            if atcg not in set_atcg:
                is_atcg = False
        if is_atcg is False:
            print(seq)
            continue

        seq_number = []

        # seq = line
        for jj in range(0, seq_size, stride):
            if jj + ngram <= len(seq):
                if word_dict is not None:
                    seq_number.append(word_dict.get(seq[jj:jj + ngram], 0))

        slice_seq_data.append(seq_number)
        slice_label_data.append(label)

        slice_index += 1

    print(slice_seq_data[0:10], len(slice_seq_data[0]))
    print(slice_label_data[0:10])

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    if len(slice_seq_data) > 0 and len(slice_label_data) > 0:
        save_dict = {
            'sequence': slice_seq_data,
            'label': slice_label_data
        }
        save_path = os.path.join(output_path, '{}_{}_gram.npz'.format(task_name, str(ngram)))
        np.savez_compressed(save_path, **save_dict)


def process_raw_text_with_annotation(sequences,
                                     annotations,
                                     labels,
                                     seq_size=1000,
                                     ngram=3,
                                     stride=1,
                                     filter_txt=None,
                                     skip_n: bool = False,
                                     word_dict: dict = None,
                                     output_path: str = './',
                                     task_name: str = 'train',
                                     gene_type_dict: dict = None):
    slice_index = 0
    slice_seq_data = []
    slice_anno_data = []
    slice_label_data = []

    set_atcg = set(list('ATCG'))

    print("seq_size: ", seq_size)

    for row in range(len(sequences)):
        seq = sequences[row]
        anno = annotations[row]
        label = labels[row]

        # print("SEQ: ", seq)
        seq = seq.upper()
        if filter_txt is not None and seq.startswith(filter_txt):
            continue

        if skip_n is True:
            if seq.find('N') > -1:
                print(seq)
                continue

        # Check if it’s not ‘ATCG’
        set_seq = set(list(seq))
        is_atcg = True
        for atcg in set_seq:
            if atcg not in set_atcg:
                is_atcg = False
        if is_atcg is False:
            print(seq)
            continue

        seq_number = []

        # seq = line
        for jj in range(0, seq_size, stride):
            if jj + ngram <= len(seq):
                if word_dict is not None:
                    seq_number.append(word_dict.get(seq[jj:jj + ngram], 0))

        slice_seq_data.append(seq_number)
        slice_label_data.append(label)

        # Sequence annotation information
        anno_len = len(anno)
        anno_position = np.zeros((len(gene_type_dict.keys()) + 2, seq_size), dtype=np.int)
        if anno_len > 0:
            for jj in range(anno_len):
                gene_type = anno[jj][0]
                gene_type = gene_type_dict.get(gene_type, 0)
                start = int(anno[jj][1])
                end = int(anno[jj][2])
                anno_position[gene_type, start:min(seq_size, end)] = 1


        slice_anno_data.append(anno_position)

        slice_index += 1

    print(slice_seq_data[0:10], len(slice_seq_data[0]))
    print(slice_anno_data[0:10])
    print(slice_label_data[0:10])

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    if len(slice_seq_data) > 0 and len(slice_label_data) > 0:
        save_dict = {
            'sequence': slice_seq_data,
            'annotation': slice_anno_data,
            'label': slice_label_data
        }
        save_path = os.path.join(output_path, '{}_{}_gram.npz'.format(task_name, str(ngram)))
        np.savez_compressed(save_path, **save_dict)


def get_epdnew_data(fasta_file: str,
                    TSS_position=5000,
                    low_position=200,
                    high_position=400,
                    ngram=3
                    ):
    sequences = []
    labels = []
    if os.path.exists(fasta_file) is False:
        return sequences, labels

    epdnew = Fasta(fasta_file)
    for epd_id in epdnew.keys():
        # Read the sequence file, the length of the sequence is 100001, and the position index of TSS is 5000.
        sequence = epdnew[epd_id]

        # Read Promoter positive example sequence, the range is [-200, +400] of TSS
        start = TSS_position - low_position
        end = TSS_position + high_position + (ngram - 1)  # ngram, Need to fill in a short sequence
        positive_seq = sequence[start:end]

        if len(positive_seq) > 0:
            sequences.append(str(positive_seq))
            labels.append(1)  # Positive example

        # Read the Promoter negative example sequence, the range is the fragment taken randomly outside of [-200, +400] of TSS
        negative_seq, _ = get_negative(sequence, TSS_position, low_position, high_position, ngram=ngram)
        if len(negative_seq) > 0:
            sequences.append(str(negative_seq))
            labels.append(0)  # Negative example

    return sequences, labels


def get_epdnew_data_with_annotation(fasta_file: str,
                                    bed_file: str,
                                    gff_file: str,
                                    TSS_position=5000,
                                    low_position=200,
                                    high_position=400,
                                    ngram=3
                                    ):
    sequences = []
    annotations = []
    labels = []
    if os.path.exists(fasta_file) is False:
        return sequences, annotations, labels

    if os.path.exists(bed_file) is False:
        return sequences, annotations, labels

    if os.path.exists(gff_file) is False:
        return sequences, annotations, labels

    # Read the TSS file, find the sequence file, and mark it manually
    promoter_df = pd.read_csv(bed_file, sep='\t',
                              header=None, names=['Ref', 'TSS', 'Location', 'V', 'Name', 'Strand'])

    epdnew = Fasta(fasta_file)

    epd_ids = []
    for item in epdnew.items():
        epd_ids.append(str(item[0]))

    promoter_df['epd_id'] = epd_ids
    promoter_df.drop('V', inplace=True, axis=1)
    promoter_df.sort_values(by=['Ref', 'Location'], ascending=[True, True], inplace=True)

    print(promoter_df.head())

    gff_file = 'data/hg38/GCF_000001405.39_GRCh38.p13_genomic.gff'

    # Annotate the signature file
    chr_gff_dict = get_refseq_gff(gff_file, include_types)

    chr_convert_dict = {}
    for k, v in chr_dict_hg38.items():
        chr_convert_dict[v] = k

    for index, row in promoter_df.iterrows():
        epd_id = row['epd_id']
        ref = row['Ref']
        location = row['Location']

        convert_ref = chr_convert_dict.get(ref, '')

        # Read the sequence file, the length of the sequence is 100001, and the position index of TSS is 5000.
        sequence = epdnew[epd_id]

        # Read Promoter positive example sequence, the range is TSS [-200, +400]
        start = TSS_position - low_position
        end = TSS_position + high_position + (ngram - 1)  # ngram, need to fill in a short sequence
        positive_seq = sequence[start:end]

        # print("chr_gff_dict: ", chr_gff_dict)
        # print("ref: ", ref)

        ann_start = location - low_position
        ann_end = location + high_position
        annotation = get_gene_feature_array(chr_gff_dict, convert_ref, ann_start, ann_end)
        if annotation is None:
            print("Row: ", row)
            annotation = []
        # print("pos annotation: ", annotation)


        if len(positive_seq) > 0:
            sequences.append(str(positive_seq))
            annotations.append(annotation)
            labels.append(1)  # Positive example

        # Read the Promoter negative example sequence, the range is the fragment taken randomly outside [-200, +400] of TSS
        negative_seq, neg_location = get_negative(sequence, TSS_position, low_position, high_position, ngram=ngram)

        ann_start = location - low_position + (neg_location - TSS_position)
        ann_end = location + high_position + (neg_location - TSS_position)
        annotation = get_gene_feature_array(chr_gff_dict, convert_ref, ann_start, ann_end)
        if annotation is None:
            print("Row: ", row)
            annotation = []

        if len(negative_seq) > 0:
            sequences.append(str(negative_seq))
            annotations.append(annotation)
            labels.append(0)  # Negative example

    return sequences, annotations, labels


if __name__ == '__main__':

    # Download from https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_genomic.gff.gz
    gff_file = '/data/hg38/GCF_000001405.39_GRCh38.p13_genomic.gff'

    ngram = 5
    stride = 1
    word_dict = get_word_dict_for_n_gram_alphabet(n_gram=ngram)

    fasta_files = ['./data/EPDnew/hg38_QMNCo.fa',
                   './data/EPDnew/hg38_oBvPw.fa',
                   './data/EPDnew/hg38_Q1zFL.fa']

    # https://epd.epfl.ch/EPD_download.php
    bed_files = ['./data/EPDnew/human_epdnew_FzZ6q.bed',
                 './data/EPDnew/human_epdnew_4hlzk.bed',
                 './data/EPDnew/human_epdnew_CAK3S.bed']

    # https://epd.epfl.ch/EPD_download.php
    task_names = ['epdnew_BOTH_Knowledge',
                  'epdnew_NO_TATA_BOX_Knowledge',
                  'epdnew_TATA_BOX_Knowledge']
    gene_type_dict = {}
    index = 1  # 0 means unknown
    for gene_type in include_types:
        gene_type_dict[gene_type] = index
        index += 1

    for fasta_file, bed_file, task_name in zip(fasta_files, bed_files, task_names):

        print("sequences: ", sequences[:10])
        print("labels: ", labels[:10])

        sequences, annotations, labels = get_epdnew_data_with_annotation(fasta_file, bed_file, gff_file, ngram=ngram)

        process_raw_text_with_annotation(sequences,
                                         annotations,
                                         labels,
                                         seq_size=600,
                                         ngram=ngram,
                                         stride=stride,
                                         filter_txt=None,
                                         skip_n=False,
                                         word_dict=word_dict,
                                         output_path='./data/{}_gram_11_knowledge'.format(ngram),
                                         task_name=task_name,
                                         gene_type_dict=gene_type_dict
                                         )
