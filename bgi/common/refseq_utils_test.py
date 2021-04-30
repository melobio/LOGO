import os

import numpy as np
from Bio import SeqIO
import argparse

# import networkx as nx
# import seaborn as sns

from bgi.common.refseq_utils import *

if __name__ == '__main__':
    ngram = 3
    word_dict_alphabet = get_word_dict_for_n_gram_alphabet(word_index_from=1, n_gram=ngram, alphabet=['A', 'G', 'C', 'T'])
    print("word_dict_alphabet: ", len(word_dict_alphabet), word_dict_alphabet)

