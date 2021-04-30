import os
import sys
from multiprocessing import Pool
import gzip
import re
import numpy as np
from pyfaidx import Fasta
import tensorflow as tf
import random
from multiprocessing import Process

sys.path.append("../../")
from bgi.common.genebank_utils import get_refseq_gff, get_gene_feature_array
from bgi.common.refseq_utils import get_word_dict_for_n_gram_alphabet

fasta = '/alldata/Hphuang_data/Genomics/CADD/GRCh37/GCF_000001405.25_GRCh37.p13_genomic.fna'
# fasta = 'E:\\Research\\Data\\Genomic\\humen\\GCF_000001405.25_GRCh37.p13_genomic.fna'

genome = Fasta(fasta)
# genome = None

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

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))

def serialize_seq_example(seq, alt_seq, alt_type, y_label):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        # 'feature': _int64_feature(x_feature),
        # 'weight': _float_feature(x_weight),
        'label': _int64_feature(y_label),
        'seq': _int64_feature(seq),
        'alt_seq': _int64_feature(alt_seq),
        'alt_type': _int64_feature(alt_type),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def create_tfrecord_with_sequence(seq_data, alt_seq_data, alt_type_data, label_data, tfrecord_file, batch_no = 1):
    """
    创建TF Record
    :param data_file:
    :param tfrecord_file:
    :return:
    """

    # if os.path.exists(data_file) is False or str(data_file).endswith('.gz') is False:
    #     return
    #
    # if os.path.exists(seq_file) is False or str(seq_file).endswith('.npz') is False:
    #     return
    #
    # loaded = np.load(seq_file, allow_pickle=True)
    # seq_data = loaded['x']
    # alt_seq_data = loaded['alt']
    # alt_type_data = loaded['type']
    #

    counter = 0
    row_index = 0

    train_file = tfrecord_file + '_{}_train.tfrecord'.format(batch_no)
    valid_file = tfrecord_file + '_{}_valid.tfrecord'.format(batch_no)
    test_file = tfrecord_file + '_{}_test.tfrecord'.format(batch_no)

    train_writer = tf.io.TFRecordWriter(train_file)
    valid_writer = tf.io.TFRecordWriter(valid_file)
    test_writer = tf.io.TFRecordWriter(test_file)
    train_samples = []
    valid_samples = []
    test_samples = []
    # with gzip.open(data_file, 'r') as pf:
    for seq, alt, type, y in zip(seq_data, alt_seq_data, alt_type_data, label_data ):
        feature = {
            'seq': tf.train.Feature(int64_list=tf.train.Int64List(value=list(seq))),
            'alt_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=list(alt))),
            'alt_type': tf.train.Feature(int64_list=tf.train.Int64List(value=list(type))),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        dice = random.random()
        if dice < 0.9:
            train_samples.append(example.SerializeToString())
            # train_writer.write(example)
        elif dice < 0.95:
            valid_samples.append(example.SerializeToString())
            # valid_writer.write(example)
        else:
            test_samples.append(example.SerializeToString())
            # test_writer.write(example)

        counter += 1
        if counter % 100000 == 0:
            print(counter)

        row_index += 1

    for example in train_samples:
        train_writer.write(example)

    for example in valid_samples:
        valid_writer.write(example)

    for example in test_samples:
        test_writer.write(example)
            # if counter > 200000:
            #     break


def process_raw_text(records: list,
                     seq_size=100,
                     ngram=3,
                     stride=1,
                     filter_txt=None,
                     skip_n: bool = False,
                     word_dict: dict = None,
                     output_path: str = './',
                     task_name: str = 'train',
                     chr_dict: dict = {},
                     gene_type_dict: dict = None,
                     padding_seq_len=500,
                     alt_shift_seq_len=10,
                     y_label=None,
                     samples=0):

    """
    SNP突变信息转换为序列信息
    :param records:
    :param seq_size:
    :param ngram:
    :param stride:
    :param filter_txt:
    :param skip_n:
    :param word_dict:
    :param output_path:
    :param task_name:
    :param chr_dict:
    :param gene_type_dict:
    :param padding_seq_len:
    :param alt_shift_seq_len:
    :return:
    """
    slice_index = 0
    slice_seq_data = []
    slice_alt_seq_data = []
    slice_alt_type_data = []
    slice_anno_data = []
    slice_label_data = []
    set_atcg = set(list('ATCG'))

    print("seq_size: ", seq_size)
    print("alt_shift_seq_len: ", alt_shift_seq_len)
    print("padding_seq_len: ", padding_seq_len)

    alphabet = {
        'N': 0,
        'A': 1,
        'G': 2,
        'C': 3,
        'T': 4,
    }

    # 染色体名称字典
    rev_chr_dict = {}
    for k, v in chr_dict.items():
        rev_chr_dict[str(v)] = k

    counter = 0
    same_counter = 0
    for record in records:
        chr = record[0]
        position = int(record[1])
        ref = record[3]
        alt = record[4]
        label = record[2]

        if y_label is not None:
            label = y_label

        ref_chr = rev_chr_dict.get(chr, '')
        if len(ref_chr) == 0:
            print(chr)
            continue

        ref_start = max((position) - padding_seq_len, 0)
        ref_end = (position + len(ref)) + padding_seq_len - len(ref) + ngram + alt_shift_seq_len

        start = max(position - padding_seq_len, 0)
        end = position + padding_seq_len + ngram + alt_shift_seq_len
        seq = str(genome[ref_chr][start:end])

        ref_left = str(genome[ref_chr][ref_start:(position-1)])
        ref_right = str(genome[ref_chr][(position + len(ref) - 1):ref_end])

        # print("Ref: ", ref)
        # print(seq.lower())
        # print(ref_left.lower() + ref.upper() + ref_right.lower())
        # print()

        #shift_seq = str(genome[ref_chr][start:end + alt_shift_seq_len+ngram])

        ref = ref.upper()
        alt = alt.upper()
        alter = alt.split(',')


        for alt in alter:
            seq = ref_left + ref + ref_right
            # alt 替换掉 ref 的内容
            alt_seq = ref_left + alt + ref_right
            # alt_seq = alt_seq[0:seq_size]

            if alt_seq == seq:
                print("seq: ", seq)
                print("alt_seq: ", alt_seq)

            seq = seq.upper()
            alt_seq = alt_seq.upper()

            if filter_txt is not None and seq.startswith(filter_txt):
                continue

            if skip_n is True:
                if seq.find('N') > -1:
                    print(seq)
                    # continue

            # 检查是否不是 ‘ATCG’的字符
            set_seq = set(list(seq))
            is_atcg = True
            for atcg in set_seq:
                if atcg not in set_atcg:
                    is_atcg = False
            if is_atcg is False:
                print(seq)
                # continue

            seq_number = []
            alt_seq_number = []

            for jj in range(0, seq_size, stride):
                if jj + ngram <= len(seq):
                    if word_dict is not None:
                        seq_number.append(word_dict.get(seq[jj:jj + ngram], 0))
                        alt_seq_number.append(word_dict.get(alt_seq[jj:jj + ngram], 0))

            if seq_number == alt_seq_number:
                same_counter = same_counter + 1
                print("ref: ", ref)
                print("alt: ", alt)
                print("seq_number: ", seq_number)
                print("alt_seq_number: ", alt_seq_number)
                print("record: ", record)
                print("same_counter: ", same_counter)
                print(seq)
                print(alt_seq)
                print()
            if len(seq_number) > 1:
                slice_seq_data.append(seq_number)
                slice_alt_seq_data.append(alt_seq_number)
                slice_label_data.append(label)

        # if counter > 0 and counter % 500000 == 0:
        #     slice_seq_data = np.array(slice_seq_data)
        #     slice_alt_seq_data = np.array(slice_alt_seq_data)
        #     alt_type = np.array(slice_seq_data == slice_alt_seq_data, dtype=np.int)
        #     alt_type = (1 - alt_type)
        #     slice_alt_seq_data = slice_alt_seq_data * alt_type
        #
        #     if os.path.exists(output_path) is False:
        #         os.makedirs(output_path)
        #     tfrecord_file = os.path.join(output_path, '{}_gram_{}_stride_{}_slice_{}'.format(task_name, str(ngram),
        #                                                                     str(stride), str(slice_index)))
        #     create_tfrecord_with_sequence(slice_seq_data, slice_alt_seq_data, alt_type, slice_alt_seq_data, tfrecord_file, batch_no = counter)
        #
        #     slice_seq_data = []
        #     slice_alt_seq_data = []
        #     alt_type = []
        #     slice_alt_seq_data = []


        ##
        # anno_len = len(anno)
        # anno_position = np.zeros((len(gene_type_dict.keys())+2, seq_size), dtype=np.int)
        # if anno_len > 0:
        #     for jj in range(anno_len):
        #         gene_type = anno[jj][0]
        #         gene_type = gene_type_dict.get(gene_type, 0)
        #
        #         start = int(anno[jj][1])
        #         end = int(anno[jj][2])
        #
        #         #print(gene_type, start, end)
        #         anno_position[gene_type, start:min(seq_size, end)] = 1
        #
        #     # print(list(anno_position))
        #     # break

        # slice_anno_data.append(anno_position)


        counter += 1
        if counter % 100000 == 0:
            print(counter)
        # slice_index += 1

    slice_index = samples

    slice_seq_data = np.array(slice_seq_data)
    slice_alt_seq_data = np.array(slice_alt_seq_data)
    slice_label_data = np.array(slice_label_data)

    alt_type = np.array(slice_seq_data == slice_alt_seq_data, dtype=np.int)
    alt_type = (1 - alt_type)
    slice_alt_seq_data = slice_alt_seq_data * alt_type

    print(slice_seq_data[0:1])
    print(slice_alt_seq_data[0:1])
    print("type: ", alt_type[0:1])
    print("label: ", slice_label_data[0:1])

    print(slice_seq_data.shape)
    print(slice_alt_seq_data.shape)
    print(alt_type.shape)

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    # if len(slice_seq_data) > 0:
    #     save_dict = {
    #         'x': slice_seq_data,
    #         'alt': slice_alt_seq_data,
    #         'type': alt_type,
    #         'label': slice_label_data
    #     }
    #     save_path = os.path.join(output_path,
    #                              '{}_gram_{}_stride_{}_slice_{}.npz'.format(task_name, str(ngram),
    #                                                                         str(stride), str(slice_index)))
    #     np.savez_compressed(save_path, **save_dict)

    if len(slice_seq_data) > 0:
        slice_seq_data = np.array(slice_seq_data)
        slice_alt_seq_data = np.array(slice_alt_seq_data)
        alt_type = np.array(slice_seq_data == slice_alt_seq_data, dtype=np.int)
        alt_type = (1 - alt_type)
        slice_alt_seq_data = slice_alt_seq_data * alt_type

        tfrecord_file = os.path.join(output_path, '{}_gram_{}_stride_{}_slice_{}'.format(task_name, str(ngram),
                                                                                         str(stride), str(slice_index)))
        create_tfrecord_with_sequence(slice_seq_data, slice_alt_seq_data, alt_type, slice_label_data, tfrecord_file,
                                      batch_no=slice_index)


def read_vcf_file(data_file, label='Benign'):
    """
    读取VCF数据
    :param data_file:
    :param label:
    :return:
    """
    records = []
    with gzip.open(data_file, 'r') as pf:
        for line in pf:
            # 去除 b' 和 \n'
            line = str(line)
            line = line[2:-3]

            tokens = line.split('\\t')
            if len(tokens) < 5:
                continue
            # 跳过第一行
            if tokens[0] == 'chr' or str(tokens[0]).startswith('#'):
                continue
            records.append(tokens)

    return records


def read_vcf_file_2(data_file, label='Benign'):
    """
    读取VCF数据
    :param data_file:
    :param label:
    :return:
    """
    records = []
    with open(data_file, 'r') as pf:
        for line in pf:
            # 去除 b' 和 \n'
            line = str(line)
            line = line[:-1]

            tokens = re.split('\\t', line)
            print(tokens)
            if len(tokens) < 5:
                continue
            # 跳过第一行
            if tokens[0] == 'chr' or str(tokens[0]).startswith('#'):
                continue

            label = tokens[2]
            if label.startswith('Benign'):
                label = 0
            else:
                label = 1

            print([tokens[0], tokens[1], label, tokens[3], tokens[4]])
            records.append([tokens[0], tokens[1], label, tokens[3], tokens[4]])

    return records


if __name__ == '__main__':
    # Read SNP information
    records = []

    ngram = 6
    seq_size = 1000
    word_dict = get_word_dict_for_n_gram_alphabet(n_gram=ngram)



    data_file = '/data/CADD/GRCh37/humanDerived_InDels.vcf.gz'
    output_file = '/data/CADD/GRCh37/output'
    results = read_vcf_file(data_file, label='')

    print("results: ", len(results))

    # Convert to ngram format
    process_raw_text(records=results,
                     seq_size=seq_size,
                     padding_seq_len=seq_size // 2,
                     ngram=ngram,
                     stride=1,
                     filter_txt=None,
                     word_dict=word_dict,
                     output_path=output_file,
                     task_name='humanDerived_InDels',
                     chr_dict=chr_dict,
                     y_label=0
                     )



    data_file = '/data/CADD/GRCh37/simulation_InDels.vcf.gz'
    output_file = '/data/CADD/GRCh37/output'
    results = read_vcf_file(data_file, label='')

    print("results: ", len(results))

    # Convert to ngram format
    process_raw_text(records=results,
                     seq_size=seq_size,
                     padding_seq_len=seq_size // 2,
                     ngram=ngram,
                     stride=1,
                     filter_txt=None,
                     word_dict=word_dict,
                     output_path=output_file,
                     task_name='simulation_InDels',
                     chr_dict=chr_dict,
                     y_label=1
                     )

    data_file = '/data/CADD/GRCh37/simulation_SNVs.vcf.gz'
    output_file = '/data/CADD/GRCh37/SNVS'

    results = read_vcf_file(data_file, label='')

    print("results: ", len(results))
    slice_size = 100000

    pool = Pool(processes=16)

    # Convert to ngram format
    for ii in range(0, len(results), slice_size):
        vcfs = results[ii:min((ii+slice_size), len(results))]
        print(ii, len(vcfs))
        pool.apply_async(process_raw_text, args=(vcfs,
                                           seq_size,
                                           ngram,
                                           1,
                                           None,
                                           False,
                                           word_dict,
                                           output_file,
                                           'simulation_SNVs',
                                           chr_dict,
                                           None,
                                           seq_size // 2,
                                           10,
                                           1,
                                           min((ii + slice_size), len(results))))

    pool.close()
    pool.join()

    data_file = '/data/CADD/GRCh37/humanDerived_SNVs.vcf.gz'
    output_file = '/data/CADD/GRCh37/SNVS'

    results = read_vcf_file(data_file, label='')

    print("results: ", len(results))

    pool = Pool(processes=16)

    # Convert to ngram format
    for ii in range(0, len(results), slice_size):
        vcfs = results[ii:min((ii + slice_size), len(results))]
        print(ii, len(vcfs))
        pool.apply_async(process_raw_text, args=(vcfs,
                                                 seq_size,
                                                 ngram,
                                                 1,
                                                 None,
                                                 False,
                                                 word_dict,
                                                 output_file,
                                                 'humanDerived_SNVs',
                                                 chr_dict,
                                                 None,
                                                 seq_size // 2,
                                                 10,
                                                 0,
                                                 min((ii + slice_size), len(results))))

    pool.close()
    pool.join()


