import os

import numpy as np
from Bio import SeqIO
import argparse

# import networkx as nx
# import seaborn as sns

atcg_dict = {
    'A': 1,
    'G': 2,
    'C': 3,
    'T': 4,
    'N': 0
}


def get_word_dict_for_n_gram_number(word_index_from=10, n_gram:int=6, alphabet:list=[0, 1, 2, 3, 4], predefined_tokens:list=[]):
    word_dict = {}

    if predefined_tokens is not None and len(predefined_tokens) > 0:
        for token in predefined_tokens:
            word_dict[token] = len(word_dict)

    word_set = []
    previous_layer_word_set = []
    add_word_set = set()
    word_index_from = max(word_index_from, len(predefined_tokens))
    for ii in range(n_gram):
        for word in alphabet:
            if ii == 0:
                word_set.append(word)
                add_word_set.add(word)
                word_dict[word] = word_index_from + len(word_dict)
            else:
                for add_word in previous_layer_word_set:
                    if len(str(add_word)) == ii:
                        new_word = add_word * 10 + word
                        if new_word in add_word_set:
                            continue
                        word_set.append(new_word)
                        add_word_set.add(new_word)
                        word_dict[new_word] = word_index_from + len(word_dict)
        previous_layer_word_set = word_set
        word_set = []
    return word_dict


def get_word_dict_for_n_gram_alphabet(word_index_from=10, n_gram:int=6, alphabet:list=['N', 'A', 'G', 'C', 'T'], predefined_tokens:list=[]):
    word_dict = {}

    if predefined_tokens is not None and len(predefined_tokens) > 0:
        for token in predefined_tokens:
            word_dict[token] = len(word_dict)

    word_set = []
    previous_layer_word_set = []
    add_word_set = set()
    word_index_from = max(word_index_from, len(predefined_tokens))
    for ii in range(n_gram):
        for word in alphabet:
            if ii == 0:
                word_set.append(word)
                add_word_set.add(word)
                word_dict[word] = word_index_from + len(word_dict)
            else:
                for add_word in previous_layer_word_set:
                    if len(str(add_word)) == ii:

                        if str(add_word).startswith('N'):
                            continue

                        new_word = add_word + '' + word

                        word_set.append(new_word)
                        add_word_set.add(new_word)
                        word_dict[new_word] = word_index_from + len(word_dict)

                        # print(word, add_word, new_word, word_dict.get(new_word, 0))

        previous_layer_word_set = word_set
        word_set = []
    return word_dict



def process_fasta(fname, c1, c2, filter_txt=None):
    genome = SeqIO.parse(fname, 'fasta')
    if filter_txt:
        chroms = [GB for GB in genome if 'NC_' in GB.id]
    else:
        chroms = [GB for GB in genome]
    genome = ''.join([i.seq.__str__() for i in chroms]).upper()
    genome_chunks = [genome[i:i + c1] for i in range(0, len(genome), c1) if
                     not 'N' in genome[i:i + c1] and set(genome[i:i + c1]) == set('ATGC')]
    clean_genome = ''.join(genome_chunks)
    data = [clean_genome[i:i + c2] for i in range(0, len(clean_genome), c2)]

    return data


def process_fasta_raw_text(fname,
                           chunk_size=10000,
                           seq_size=1000,
                           seq_stride=500,
                           ngram=3,
                           stride=1,
                           filter_txt=None,
                           skip_n: bool = False,
                           word_dict: dict = None,
                           slice_size: int = 100000,
                           output_path: str = './',
                           hg_name: str = 'hg19'):

    index = 0
    chunks = ''

    # 例如 3-gram， 取999，存在问题， 只能取到997个特征
    seq_size = max(seq_size, seq_size//ngram * ngram + (ngram-1))

    slice_index = 0

    slice_seq_raw_data = []
    slice_seq_num_data = []

    print("seq_size: ", seq_size)
    with open(fname, mode='r', encoding='utf-8') as f:
        for line in f:
            # 去除 '\n'
            line = line[:-1]
            line = line.upper()
            if filter_txt is not None and line.startswith(filter_txt):
                continue

            if line.find('N') > -1 and skip_n is True:
                continue

            if len(chunks) < chunk_size:
                chunks += line
            else:
                for ii in range(0, chunk_size, seq_stride):
                    if ii + seq_size <= chunk_size:
                        seq = chunks[ii:int(ii+seq_size)]
                        seq_list = []
                        seq_number = []
                        for jj in range(0, len(seq), stride):
                            if jj + ngram <= len(seq):
                                if word_dict is not None:
                                    seq_number.append(word_dict.get(seq[jj:jj+ngram], 0))
                                #seq_list.append(seq[jj:jj+ngram])

                        #slice_seq_raw_data.append(seq_list)
                        # print(seq_number)
                        slice_seq_num_data.append(seq_number)
                        slice_index += 1

                        if slice_index > 0 and slice_index % slice_size == 0:
                            save_dict = {
                                'data': slice_seq_num_data,
                                #'dict': word_dict
                            }
                            save_path = os.path.join(output_path, '{}_seq_gram_{}_stride_{}_slice_{}.npz'.format(hg_name, str(ngram), str(stride), str(slice_index)))
                            np.savez_compressed(save_path, **save_dict)

                            slice_seq_raw_data = []
                            slice_seq_num_data = []

                # 更新为剩余的序列
                chunks = chunks[ii:]
                chunks += line

            index += 1
            if index % 100000 == 0:
                print(index, len(line))

        if len(slice_seq_num_data) > 0:
            save_dict = {
                'data': slice_seq_num_data,
                'dict': word_dict
            }
            save_path = os.path.join(output_path,
                                     '{}_seq_gram_{}_stride_{}_slice_{}.npz'.format(hg_name, str(ngram),
                                                                                    str(stride), str(slice_index)))
            np.savez_compressed(save_path, **save_dict)
            slice_seq_raw_data = []
            slice_seq_num_data = []


if __name__ == '__main__':

    _argparser = argparse.ArgumentParser(
        description='A data preprocessing of the Transformer language model in Genomics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--data', type=str, required=True, metavar='PATH',
        help='A path of hg19/38 file.')
    _argparser.add_argument(
        '--output', type=str, default='transformer_gene', metavar='NAME',
        help='A path which save processed file')
    _argparser.add_argument(
        '--chunk-size', type=int, default=10000, metavar='INTEGER',
        help='chunk size')
    _argparser.add_argument(
        '--seq-size', type=int, default=1000, metavar='INTEGER',
        help='Sequence size')
    _argparser.add_argument(
        '--seq-stride', type=int, default=100, metavar='INTEGER',
        help='Sequence stride size')
    _argparser.add_argument(
        '--ngram', type=int, default=3, metavar='INTEGER',
        help='NGram')
    _argparser.add_argument(
        '--stride', type=int, default=1, metavar='INTEGER',
        help='Stride')
    _argparser.add_argument(
        '--slice-size', type=int, default=100000, metavar='INTEGER',
        help='Slice size')
    _argparser.add_argument(
        '--hg-name', type=str, default='hg19', metavar='NAME',
        help='hg name')




    _args = _argparser.parse_args()

    data_path = _args.data
    output_path = _args.output
    chunk_size = _args.chunk_size
    seq_size = _args.seq_size
    seq_stride = _args.seq_stride
    ngram = _args.ngram
    stride = _args.stride
    slice_size = _args.slice_size
    hg_name = _args.hg_name

    word_dict_alphabet = get_word_dict_for_n_gram_alphabet(n_gram=ngram)
    print("word_dict_alphabet: ", len(word_dict_alphabet))

    word_dict_number = get_word_dict_for_n_gram_number(n_gram=ngram)
    print("word_dict_number: ", len(word_dict_number))


    process_fasta_raw_text(data_path,
                           chunk_size=chunk_size,
                           seq_size=seq_size,
                           seq_stride=seq_stride,
                           ngram=ngram,
                           stride=stride,
                           filter_txt='>NC_',
                           word_dict=word_dict_alphabet,
                           slice_size=slice_size,
                           output_path=output_path,
                           hg_name=hg_name)

    # data_path = 'D:\\Genomics\\Data\\Hg38\\GCF_000001405.25_GRCh37.p13_genomic.fna'
    #
    # output_path = 'D:\\Genomics\\Data\\Hg38\\hg19\\'
    # chunk_size = 10000
    # seq_size = 1000
    # seq_stride = 100
    # ngram = 3
    # stride = 1
    # slice_size = 100000
    # hg_name = 'hg19'
    #
    # word_dict_alphabet = get_word_dict_for_n_gram_alphabet(n_gram=ngram)
    # print("word_dict_alphabet: ", len(word_dict_alphabet))
    #
    # process_fasta_raw_text(data_path,
    #                        chunk_size=chunk_size,
    #                        seq_size=seq_size,
    #                        seq_stride=seq_stride,
    #                        ngram=ngram,
    #                        stride=stride,
    #                        filter_txt='>NC_',
    #                        word_dict=word_dict_alphabet,
    #                        slice_size=slice_size,
    #                        output_path=output_path,
    #                        hg_name=hg_name)
