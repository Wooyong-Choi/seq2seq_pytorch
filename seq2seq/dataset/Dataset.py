import re
import sys
from operator import itemgetter

from tqdm import tqdm
from torch.utils.data import Dataset

import Vocab


class Dataset(Dataset):
    """
    A dataset basically supports iteration over all the examples it contains.
    We currently supports only text data with this class.
    This class is inheriting Dataset class in torch.utils.data.
    """

    def __init__(self, src_file_path, tgt_file_path, max_src_len, max_tgt_len,
                 max_cut=False, src_vocab_size=sys.maxsize, tgt_vocab_size=sys.maxsize):
        super(Dataset, self).__init__()
        
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        self.max_cut = max_cut
        
        self.src_vocab = Vocab()
        self.tgt_vocab = Vocab()
        
        self.train_pairs = None
        self.test_pairs = None
        
        self._prepareData()
        
    def __getitem__(self, index):
        src_indice = self.src_vocab.sentence2indice(self.train_pairs[index][0])
        tgt_indice = self.tgt_vocab.sentence2indice(self.train_pairs[index][1])
        src_length = len(self.train_pairs[index][0])
        tgt_length = len(self.train_pairs[index][1])
        return (src_indice, tgt_indice, src_length, tgt_length)
                
    def __len__(self):
        return len(self.train_pairs)
    
    def _prepareData(self):
        pairs = self._readData()
        print("Read %s sentence pairs" % len(pairs))
        print()
        
        pairs = self._filterPairs(pairs)
        print("Trim data to %s sentence pairs" % len(pairs))
        print("Avg length of src : ", sum([len(pair[0]) for pair in pairs]) / len(pairs))
        print("Avg length of tgt : ", sum([len(pair[1]) for pair in pairs]) / len(pairs))
        print()
        
        self._prepareVocab(pairs)
        print()
        
        self.train_pairs = pairs[:int(len(pairs)*0.9)]
        self.test_pairs = pairs[int(len(pairs)*0.9):]
        
        print("Success to preprocess data!")
        print()
    
    def _readData(self):
        print("Reading lines...")
    
        # Read the file and split into lines
        src_lines = open(self.src_file_path, 'r', encoding='utf-8').readlines()
        tgt_lines = open(self.tgt_file_path, 'r', encoding='utf-8').readlines()
        
        # Split every line into pairs and normalize
        pairs = [[src_lines[i].strip(), tgt_lines[i].strip()] for i in range(len(src_lines))]
        
        return pairs
    
    def _filterPairs(self, pairs):
        if self.max_cut:
            new_pairs = [[pair[0].lower().split(' ')[:self.max_src_len],
                          pair[1].lower().split(' ')[:self.max_tgt_len]] for pair in pairs]
        else:
            new_pairs = [[pair[0].lower().split(' '), pair[1].lower().split(' ')] for pair in pairs]
            new_pairs = [pair for pair in new_pairs if self._filterPair(pair)]
        
        
        return new_pairs
        
    def _filterPair(self, p):
        return (len(p[0]) <= self.max_src_len and len(p[1]) <= self.max_tgt_len and
                len(p[0]) > 0 and len(p[1]) > 0)
    
    def _prepareVocab(self, pairs):
        for pair in pairs:
            self.src_vocab.addSentence(pair[0])
            self.tgt_vocab.addSentence(pair[1])
            
        org_src_n_words = self.src_vocab.n_words
        org_tgt_n_words = self.tgt_vocab.n_words
            
        self.src_vocab.makeVocabDict(self.src_vocab_size)
        self.tgt_vocab.makeVocabDict(self.tgt_vocab_size)
        
        self.src_vocab_size = self.src_vocab.n_words
        self.tgt_vocab_size = self.tgt_vocab.n_words
        
        print("Source vocab : {} ({} reduced)".format(self.src_vocab.n_words, org_src_n_words - self.src_vocab.n_words))
        print("Target vocab : {} ({} reduced)".format(self.tgt_vocab.n_words, org_tgt_n_words - self.tgt_vocab.n_words))
                                                              
def sorted_collate_fn(batch):
    """
    Sort data in decreasing order of source sentences lengths
    for packing padded sequence
    """
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    src_indices = []
    tgt_indices = []
    src_lengths = []
    tgt_lengths = []
    for item in batch:
        src_indices.append(item[0])
        tgt_indices.append(item[1])
        src_lengths.append(item[2])
        tgt_lengths.append(item[3])
    return [src_indices, tgt_indices, src_lengths, tgt_lengths]