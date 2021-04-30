# import vcf
import gzip
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pyfaidx import Fasta

sys.path.append("../../")
from bgi.common.genebank_utils import get_gene_feature_array, get_refseq_gff

fasta = 'D:\\Genomics\\Data\\Hg38\\GCF_000001405.25_GRCh37.p13_genomic.fna'
fasta = '/data/huadajiyin/data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna'
# fasta = '/alldata/Hphuang_data/Genomics/GCF_000001405.25_GRCh37.p13_genomic.fna'
genome = Fasta(fasta)

# 染色体, 编号
chr_dict = {"NC_000001.10": 1,
            "NC_000002.11": 2,
            "NC_000003.11": 3,
            "NC_000004.11": 4,
            "NC_000005.9": 5,
            "NC_000006.11": 6,
            "NC_000007.13": 7,
            "NC_000008.10": 8,
            "NC_000009.11": 9,
            "NC_000010.10": 10,
            "NC_000011.9": 11,
            "NC_000012.11": 12,
            "NC_000013.10": 13,
            "NC_000014.8": 14,
            "NC_000015.9": 15,
            "NC_000016.9": 16,
            "NC_000017.10": 17,
            "NC_000018.9": 18,
            "NC_000019.9": 19,
            "NC_000020.10": 20,
            "NC_000021.8": 21,
            "NC_000022.10": 22,
            "NC_000023.10": 'X',
            "NC_000024.9": 'Y'}

# 注释类型
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



def vcf_reader(filename:str, chr_dict:dict, delimiter='\t'):
    """
    读取VCF文件，并转换CHR 为 'NC_'格式

    :param filename:
    :param chr_dict:
    :param delimiter:
    :return:
    """

    # CHROM	POS ID	REF	ALT	INFO
    chr_convert_dict = {}
    for k, v in chr_dict.items():
        chr_convert_dict[str(v)] = str(k)

    records = []
    with open(filename, mode='r', encoding='utf-8') as f:
        for line in f:
            # Delete '\n'
            line = line[:-1]

            tokens = line.split(delimiter)
            if len(tokens) > 0:
                rev_chr = chr_convert_dict.get(tokens[0], '')
                tokens.append(rev_chr)
                records.append(tokens)

    return records


def vcf_gz_reader(filename:str, chr_dict:dict, delimiter='\\t'):
    """
    读取VCF文件，并转换CHR 为 'NC_'格式

    :param filename:
    :param chr_dict:
    :param delimiter:
    :return:
    """

    # CHROM	POS ID	REF	ALT	INFO
    chr_convert_dict = {}
    for k, v in chr_dict.items():
        chr_convert_dict[str(v)] = str(k)

    records = []
    #with open(filename, mode='r', encoding='utf-8') as f:
    with gzip.open(filename, 'r') as pf:
        for line in pf:
            line = str(line)
            # 去除 b' 和 \n'
            line = line[2:-3]

            tokens = line.split(delimiter)
            if len(tokens) > 0:
                rev_chr = chr_convert_dict.get(tokens[0], '')
                tokens.append(rev_chr)
                records.append(tokens)

    return records



def extract_variant_seq(loc, orient, chr, chr_gff_dict, name='', padding_seq_len=500, alt_shift_seq_len=50):
    """
    提取突变的序列，以及1000bp周围的标注
    :param loc:
    :param orient:
    :param genome:
    :param ref:
    :param chr_gff_dict:
    :param name:
    :return:
    """
    start = int(loc) - padding_seq_len
    end = int(loc) + padding_seq_len

    if genome is None:
        return None, None, None

    seq = str(genome[chr][max(start,0):end])

    alt_seq = str(genome[chr][max(start,0):(end + alt_shift_seq_len)])
    alt_seq = alt_seq.upper()
    set_atcg = set(list('ATCG'))

    # if orient == '-':
    #     reverse_seq = Seq(promoter, generic_dna)
    #     promoter = str(reverse_seq.reverse_complement())

    seq = seq.upper()

    # 检查是否不是 ‘ATCG’的字符
    set_seq = set(list(seq))
    is_atcg = True
    for atcg in set_seq:
        if atcg not in set_atcg:
            is_atcg = False
    if is_atcg is False:
        return None, None, None

    annotations = get_gene_feature_array(chr_gff_dict, chr, start, end)

    if len(annotations) == 0:
        # print("---", chr, start, end, loc, name)
        annotations = []

    if not 'N' in seq:
        return seq, alt_seq, annotations   # 和Negative_sample保持一致
    else:
        return None, None, None


def variant_to_seq(data, chr, chr_gff_dict, convert_chr=None):
    """
    从染色体TSS提取出序列，人工标准
    :param df:
    :param ref:
    :param genome:
    :param chr_gff_dict:
    :param convert_ref:
    :return:
    """
    #data = df[df.Chr == chr].copy()
    sequences = []
    alt_sequences = []
    annotations = []
    refs = []
    alts = []
    labels = []

    if len(data) > 0:
        for index, row in data.iterrows():
            if convert_chr is None:
                seq, alt_seq, anno = extract_variant_seq(row['Location'], '', chr, chr_gff_dict)
            else:
                seq, alt_seq, anno = extract_variant_seq(row['Location'], '', convert_chr, chr_gff_dict)

            if seq is not None:
                sequences.append(seq)
                alt_sequences.append(alt_seq)
                annotations.append(anno)
                refs.append(row['Ref'])
                alts.append(row['Alt'])
                labels.append(row['Sig'])

    return sequences, alt_sequences, annotations, refs, alts, labels


def extract_slice_variant_seq(data, chr, chr_gff_dict, convert_chr=None):
    sequences = []
    alt_sequences = []
    annotations = []
    refs = []
    alts = []
    labels = []

    for index, row in data.iterrows():
        if convert_chr is None:
            seq, alt_seq, anno = extract_variant_seq(row['Location'], '', chr, chr_gff_dict)
        else:
            seq, alt_seq, anno = extract_variant_seq(row['Location'], '', convert_chr, chr_gff_dict)

        if seq is not None:
            sequences.append(seq)
            alt_sequences.append(alt_seq)
            annotations.append(anno)
            refs.append(row['Ref'])
            alts.append(row['Alt'])
            labels.append(row['Sig'])

    print("Finish: ", len(sequences))
    return sequences, alt_sequences, annotations, refs, alts, labels


def extract_slice_variant_seq_and_save(data, chr, chr_gff_dict, convert_chr=None, output_path='', name='', slice_index=0):
    sequences = []
    alt_sequences = []
    annotations = []
    refs = []
    alts = []
    labels = []

    for index, row in data.iterrows():
        if convert_chr is None:
            seq, alt_seq, anno = extract_variant_seq(row['Location'], '', chr, chr_gff_dict)
        else:
            seq, alt_seq, anno = extract_variant_seq(row['Location'], '', convert_chr, chr_gff_dict)

        if seq is not None:
            sequences.append(seq)
            alt_sequences.append(alt_seq)
            annotations.append(anno)
            refs.append(row['Ref'])
            alts.append(row['Alt'])
            labels.append(row['Sig'])

    print("Finish-1: ", len(sequences))

    seq_list = []
    alt_seq_list = []
    anno_list = []
    ref_list = []
    alt_list = []
    label_list = []

    counter = 0
    for ii in range(len(sequences)):
        seq = sequences[ii]
        alt_seq = alt_sequences[ii]
        anno = annotations[ii]
        ref = refs[ii]
        alt = alts[ii]
        label = labels[ii]
        if len(seq) > 0:
            seq_list.append(seq)
            alt_seq_list.append(alt_seq)
            anno_list.append(np.array(anno))
            ref_list.append(ref)
            alt_list.append(alt)
            label_list.append(label)
            counter += 1

    print("Finish 2: ")
    snp_seqs = np.array(seq_list)
    alt_snp_seqs = np.array(seq_list)
    snp_annos = np.array(anno_list)
    snp_refs = np.array(ref_list)
    snp_alts = np.array(alt_list)
    snp_labels = np.array(label_list)

    print("Finish 3: ")

    print("SNP: ", snp_seqs.shape)
    print("Annotation: ", snp_annos.shape)
    save_dict = {
        'Sequence': snp_seqs,
        'Alt_Sequence': alt_snp_seqs,
        'Annotation': snp_annos,
        'Ref': snp_refs,
        'Alt': snp_alts,
        'Target': snp_labels
    }

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    print("Finish 4: ")
    file_name = os.path.join(output_path, '{}_chr_{}_slice_{}_sequences_{}.npz'.format(name, chr, slice_index, counter))
    np.savez_compressed(file_name, **save_dict)
    print("Save: ", file_name)

    return sequences, alt_sequences, annotations, refs, alts, labels

def variant_to_seq_parallel(data, chr, chr_gff_dict, convert_chr=None, pool_size=4, slice_size=10000, output_path='', name=''):
    """
    从染色体TSS提取出序列，人工标准
    :param df:
    :param ref:
    :param genome:
    :param chr_gff_dict:
    :param convert_ref:
    :return:
    """
    if len(data) == 0:
        return

    data_list = []
    results = []
    pool = Pool(processes=pool_size)

    for ii in range(0, len(data) // slice_size + 1):
        if ii * slice_size >= len(data):
            break
        data_slice = data[(ii * slice_size):min(((ii+1) * slice_size), len(data))].copy()
        print("Begin: ", chr, (ii * slice_size), min(((ii+1) * slice_size), len(data)), len(data))
        result = pool.apply_async(extract_slice_variant_seq_and_save,
                                  args=(
                                      data_slice,
                                      chr,
                                      chr_gff_dict,
                                      convert_chr,
                                      output_path,
                                      name,
                                      min(((ii + 1) * slice_size), len(data))
                                  )
                                  )
        results.append(result)
        print("Finish: ", chr)
    pool.close()
    pool.join()



def get_variant_sequences(clinvar_df, chr_gff_dict, chr_convert_dict, pool_size=4, output_path='', name='snp', slice_size=10000):
    """
    从突变获取相关的序列和标注信息
    :param clinvar_df:
    :param chr_gff_dict:
    :param chr_convert_dict:
    :param pool_size:
    :param output_path:
    :return:
    """
    seq_outs = []
    alt_seq_outs = []
    anno_outs = []
    ref_outs = []
    alt_outs = []
    label_outs = []

    pool = Pool(processes=pool_size)
    results = []
    for chr in clinvar_df.Chr.unique():
        convert_chr = chr_convert_dict.get(chr, '')
        if len(convert_chr) == 0:
            continue

        print("chr: ", chr, convert_chr)
        data = clinvar_df[clinvar_df.Chr == chr].copy()
        for ii in range(0, len(data) // slice_size + 1):
            if ii * slice_size >= len(data):
                break
            data_slice = data[(ii * slice_size):min(((ii+1) * slice_size), len(data))].copy()
            print("Begin: ", chr, (ii * slice_size), min(((ii+1) * slice_size), len(data)), len(data))
            result = pool.apply_async(extract_slice_variant_seq_and_save,
                                      args=(
                                          data_slice,
                                          chr,
                                          chr_gff_dict,
                                          convert_chr,
                                          output_path,
                                          name,
                                          min(((ii + 1) * slice_size), len(data))
                                      )
                                      )


    pool.close()
    pool.join()

    # for result in results:
    #     seq, alt_seq, anno, ref, alt, label = result.get()
    #     if len(seq) > 0:
    #         seq_outs.extend(seq)
    #         alt_seq_outs.extend(alt_seq)
    #         anno_outs.extend(anno)
    #         ref_outs.extend(ref)
    #         alt_outs.extend(alt)
    #         label_outs.extend(label)
    #
    # print(len(seq_outs))
    # print(len(anno_outs))
    #
    # seq_list = []
    # alt_seq_list = []
    # anno_list = []
    # ref_list = []
    # alt_list = []
    # label_list = []
    #
    # counter = 0
    # for ii in range(len(seq_outs)):
    #     seqs = seq_outs[ii]
    #     alt_seqs = alt_seq_outs[ii]
    #     annos = anno_outs[ii]
    #     refs = ref_outs[ii]
    #     alts = alt_outs[ii]
    #     labels = label_outs[ii]
    #     for jj in range(len(seqs)):
    #         if len(seqs[jj]) > 0:
    #             seq_list.append(seqs[jj])
    #             alt_seq_list.append(alt_seqs[jj])
    #             anno_list.append(np.array(annos[jj]))
    #             ref_list.append(refs)
    #             alt_list.append(alts)
    #             label_list.append(labels)
    #             counter += 1
    #
    # snp_seqs = np.array(seq_list)
    # alt_snp_seqs = np.array(seq_list)
    # snp_annos = np.array(anno_list)
    # snp_refs = np.array(ref_list)
    # snp_alts = np.array(alt_list)
    # snp_labels = np.array(label_list)
    #
    # print("SNP: ", snp_seqs.shape)
    # print("Annotation: ", snp_annos.shape)
    # save_dict = {
    #     'Sequence': snp_seqs,
    #     'Alt_Sequence': alt_snp_seqs,
    #     'Annotation': snp_annos,
    #     'Ref': snp_refs,
    #     'Alt': snp_alts,
    #     'Target': snp_labels
    # }
    #
    # if os.path.exists(output_path) is False:
    #     os.makedirs(output_path)
    #
    # file_name = os.path.join(output_path, '{}_variant_sequences_{}.npz'.format(name, counter))
    # np.savez_compressed(file_name, **save_dict)

def get_variant_sequences_parallel_by_chr(clinvar_df, chr_gff_dict, chr_convert_dict, pool_size=4, output_path='', name='snp'):
    """
    从突变获取相关的序列和标注信息
    :param clinvar_df:
    :param chr_gff_dict:
    :param chr_convert_dict:
    :param pool_size:
    :param output_path:
    :return:
    """
    for chr in clinvar_df.Chr.unique():
        convert_chr = chr_convert_dict.get(chr, '')
        if len(convert_chr) == 0:
            continue

        print("chr: ", chr, convert_chr)
        data = clinvar_df[clinvar_df.Chr == chr].copy()
        variant_to_seq_parallel(data=data, chr=chr, chr_gff_dict=chr_gff_dict, convert_chr=convert_chr, output_path=output_path, name=name, pool_size=pool_size)
        print("Finish: ", chr)



if __name__ == '__main__':

    # fasta = 'D:\\Genomics\\Data\\Hg38\\GCF_000001405.25_GRCh37.p13_genomic.fna'
    # fasta = '/data/huadajiyin/data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna'
    # genome = Fasta(fasta)

    data_file = 'D:\\Genomics\\Data\\CADD\\validation\\clinvar_20180729_pathogenic_all_GRCh37.vcf'
    data_file = '/alldata/Hphuang_data/Genomics/CADD/GRCh37/simulation_SNVs.vcf.gz'
    # data_file = 'D:\\Genomics\\Data\\CADD\\GRCh37\\simulation_SNVs.vcf.gz'
    data_file = '/data/huadajiyin/data/hg19/humanDerived_InDels.vcf.gz'
    data_file = '/data/huadajiyin/data/CADD/validation/clinvar_20180729_pathogenic_all_GRCh37.vcf'

    # gff_file = 'F:\\Research\\Genomics\\humen51\\GCF_000001405.25_GRCh37.p13_genomic.gff'
    gff_file = '/alldata/Hphuang_data/Genomics/GCF_000001405.25_GRCh37.p13_genomic.gff'
    gff_file = '/data/huadajiyin/data/hg19/GCF_000001405.25_GRCh37.p13_genomic.gff'

    output = '/alldata/Hphuang_data/Genomics/CADD/GRCh37/all'
    output = '/data/huadajiyin/data/CADD/GRCh37/All'
    output = '/data/huadajiyin/data/CADD/validation/All'
    name = 'clinvar_20180729_pathogenic_all_GRCh37'

    records = vcf_reader(filename=data_file, chr_dict=chr_dict)

    variant_df = pd.DataFrame(records, columns=['Chr', 'Location', 'V', 'Ref', 'Alt', 'Rev_Chr'])
    variant_df['Sig'] = 'Pathogenic'
    variant_df.sort_values(by=['Chr', 'Location'], ascending=[True, True], inplace=True)

    print(variant_df.head())


    # 标注特征文件
    chr_gff_dict = get_refseq_gff(gff_file, include_types)

    chr_convert_dict = {}
    for k, v in chr_dict.items():
        chr_convert_dict[str(v)] = str(k)

    # get_variant_sequences_parallel_by_chr(variant_df, chr_gff_dict, chr_convert_dict, pool_size=80, output_path=output, name=name)
    get_variant_sequences(variant_df, chr_gff_dict, chr_convert_dict, pool_size=80, output_path=output, name=name)


