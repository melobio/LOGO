import argparse
import os
import gc
from multiprocessing import Pool
import sys
import time

import numpy as np
from Bio import SeqIO
from Bio.Alphabet import generic_dna
from Bio.Seq import Seq

sys.path.append("../")
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number, get_word_dict_for_n_gram_alphabet

atcg_dict = {
    'A': 1,
    'G': 2,
    'C': 3,
    'T': 4,
    'N': 0
}


def preccess_seq_chunks(seq_chunks, slice_index, seq_size, seq_stride, stride, ngram, word_dict, output_path, hg_name):
    set_atcg = set(list('ATCG'))
    slice_seq_num_data = []
    for ii in range(0, len(seq_chunks), seq_stride):
        if ii + seq_size <= len(seq_chunks):
            seq = seq_chunks[ii:int(ii + seq_size)]
            seq_number = []
            for jj in range(0, len(seq), stride):
                if jj + ngram <= len(seq):
                    if word_dict is not None:
                        seq_number.append(word_dict.get(seq[jj:jj + ngram], 0))

            slice_seq_num_data.append(seq_number)

            my_dna = Seq(seq, generic_dna)
            seq_flip = str(my_dna.reverse_complement())
            seq_list = []
            seq_number = []
            for jj in range(0, len(seq_flip), stride):
                if jj + ngram <= len(seq_flip):
                    if word_dict is not None:
                        seq_number.append(word_dict.get(seq_flip[jj:jj + ngram], 0))
                    # seq_list.append(seq[jj:jj+ngram])
            slice_seq_num_data.append(seq_number)

    save_dict = {
        'data': slice_seq_num_data,
    }
    save_path = os.path.join(output_path,
                             '{}_seq_gram_{}_stride_{}_slice_{}.npz'.format(hg_name, str(ngram),
                                                                            str(stride),
                                                                            str(slice_index)))
    np.savez_compressed(save_path, **save_dict)
    time.sleep(5)


def process_fasta_raw_text_parallel(fname,
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
                                    hg_name: str = 'hg19',
                                    pool_size: int = 8):
    index = 0
    chunks = ''

    # Sequence size
    seq_size = max(seq_size, seq_size // ngram * ngram + (ngram - 1))

    slice_index = 0

    slice_seq_raw_data = []
    slice_seq_num_data = []
    set_atcg = set(list('ATCG'))

    print("seq_size: ", seq_size)
    with open(fname, mode='r', encoding='utf-8') as f:

        pool = Pool(processes=pool_size)

        for line in f:
            # Delete '\n'
            line = line[:-1]
            line = line.upper()
            if filter_txt is not None and line.startswith(filter_txt):
                continue

            if skip_n is True:
                if line.find('N') > -1:
                    continue

            # Check if it’s not ‘ATCG’
            set_seq = set(list(line))
            is_atcg = True
            for atcg in set_seq:
                if atcg not in set_atcg:
                    is_atcg = False
            if is_atcg is False:
                continue

            if len(chunks) < (seq_stride * slice_size + (seq_size - seq_stride)):
                chunks += line
            else:
                for ii in range(0, int(seq_stride * slice_size), seq_stride):
                    if ii + seq_size <= (seq_stride * slice_size + (seq_size - seq_stride)):
                        continue

                print("ii: ", slice_index, seq_stride * slice_size, len(chunks))
                print("chunks: ", len(chunks))

                seq_chunks = chunks[:(ii + seq_size)]
                chunks = chunks[(ii + seq_size):]
                chunks += line

                slice_index += int((ii + seq_stride) / seq_stride)
                pool.apply_async(preccess_seq_chunks, args=(
                seq_chunks, slice_index, seq_size, seq_stride, stride, ngram, word_dict, output_path, hg_name))

            index += 1
            if index % 100000 == 0:
                print(index, len(line))

        if len(chunks) > 0:
            slice_index += int(len(chunks) / seq_stride)
            pool.apply_async(preccess_seq_chunks,
                             args=(chunks, slice_index, seq_size, seq_stride, stride, ngram, word_dict, output_path,
                                   hg_name))

        pool.close()
        pool.join()


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

    # Sequence length
    seq_size = max(seq_size, seq_size // ngram * ngram + (ngram - 1))

    slice_index = 0

    slice_seq_raw_data = []
    slice_seq_num_data = []

    set_atcg = set(list('ATCG'))

    print("seq_size: ", seq_size)
    with open(fname, mode='r', encoding='utf-8') as f:
        for line in f:
            # Delete '\n'
            line = line[:-1]
            line = line.upper()
            if filter_txt is not None and line.startswith(filter_txt):
                continue

            if skip_n is True:
                if line.find('N') > -1:
                    continue

            # Check if it’s not ‘ATCG’
            set_seq = set(list(line))
            is_atcg = True
            for atcg in set_seq:
                if atcg not in set_atcg:
                    is_atcg = False
            if is_atcg is False:
                continue

            if len(chunks) < chunk_size:
                chunks += line
            else:
                for ii in range(0, chunk_size, seq_stride):
                    if ii + seq_size <= chunk_size:
                        seq = chunks[ii:int(ii + seq_size)]

                        # Check if it’s not ‘ATCG’
                        set_seq = set(list(seq))
                        is_atcg = True
                        for atcg in set_seq:
                            if atcg not in set_atcg:
                                is_atcg = False
                        if is_atcg is False:
                            continue

                        seq_list = []
                        seq_number = []
                        for jj in range(0, len(seq), stride):
                            if jj + ngram <= len(seq):
                                if word_dict is not None:
                                    seq_number.append(word_dict.get(seq[jj:jj + ngram], 0))

                        slice_seq_num_data.append(seq_number)

                        my_dna = Seq(seq, generic_dna)
                        seq_flip = str(my_dna.reverse_complement())
                        seq_list = []
                        seq_number = []
                        for jj in range(0, len(seq_flip), stride):
                            if jj + ngram <= len(seq_flip):
                                if word_dict is not None:
                                    seq_number.append(word_dict.get(seq_flip[jj:jj + ngram], 0))

                        slice_seq_num_data.append(seq_number)
                        slice_index += 1

                        if slice_index > 0 and slice_index % slice_size == 0:
                            save_dict = {
                                'data': slice_seq_num_data,
                                # 'dict': word_dict
                            }
                            save_path = os.path.join(output_path,
                                                     '{}_seq_gram_{}_stride_{}_slice_{}.npz'.format(hg_name, str(ngram),
                                                                                                    str(stride),
                                                                                                    str(slice_index)))
                            np.savez_compressed(save_path, **save_dict)

                            slice_seq_raw_data = []
                            slice_seq_num_data = []

                # Update to the remaining sequence
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
            del slice_seq_raw_data
            del slice_seq_num_data


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
    _argparser.add_argument(
        '--pool-size', type=int, default=4, metavar='INTEGER',
        help='Pool size')

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
    pool_size = _args.pool_size

    word_dict_alphabet = get_word_dict_for_n_gram_alphabet(n_gram=ngram, word_index_from=10)
    print("word_dict_alphabet: ", len(word_dict_alphabet))

    word_dict_number = get_word_dict_for_n_gram_number(n_gram=ngram, word_index_from=10)
    print("word_dict_number: ", len(word_dict_number))

    process_fasta_raw_text_parallel(data_path,
                                    chunk_size=chunk_size,
                                    seq_size=seq_size,
                                    seq_stride=seq_stride,
                                    ngram=ngram,
                                    stride=stride,
                                    filter_txt='>NC_',
                                    word_dict=word_dict_alphabet,
                                    slice_size=slice_size,
                                    output_path=output_path,
                                    hg_name=hg_name,
                                    skip_n=True,
                                    pool_size=pool_size)

