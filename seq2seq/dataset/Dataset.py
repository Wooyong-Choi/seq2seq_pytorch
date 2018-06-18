import re
import sys
from operator import itemgetter

from tqdm import tqdm
from torch.utils.data import Dataset

from .Vocab import Vocab


# TODO: 너무 느림..
class Dataset(Dataset):
    """
    A dataset basically supports iteration over all the examples it contains.
    We currently supports only text data with this class.
    This class is inheriting Dataset class in torch.utils.data.
    """

    def __init__(self, src_file_path, tgt_file_path, max_length, max_cut=False, src_vocab_size=sys.maxsize, tgt_vocab_size=sys.maxsize):
        super(Dataset, self).__init__()
        
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path
        
        self.max_length = max_length
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        self.pairs = None
        self.src_vocab = Vocab()
        self.tgt_vocab = Vocab()
        
        self.train_pairs = None
        self.test_pairs = None
        
        self._prepareData(max_cut=max_cut)
        
    def __getitem__(self, index):
        src_indices = self.src_vocab.sentence_to_indices(self.train_pairs[index][0])
        tgt_indices = self.tgt_vocab.sentence_to_indices(self.train_pairs[index][1])
        src_lengths = len(self.train_pairs[index][0])
        tgt_lengths = len(self.train_pairs[index][1])
        return (src_indices, tgt_indices, src_lengths, tgt_lengths)
                
    def __len__(self):
        return len(self.train_pairs)
    
    def _prepareData(self, max_cut):
        self._readData()
        print("Read %s sentence pairs" % len(self.pairs))
        
        self._filterPairs(max_cut)
        print("Trim data to %s sentence pairs" % len(self.pairs))
        
        self._prepareVocab()
        
        self.train_pairs = self.pairs[:int(len(self.pairs)*0.8)]
        self.test_pairs = self.pairs[int(len(self.pairs)*0.8):]
    
    def _readData(self, reverse=False):
        print("Reading lines...")
    
        # Read the file and split into lines
        src_lines = open(self.src_file_path, 'r', encoding='utf-8').readlines()
        tgt_lines = open(self.tgt_file_path, 'r', encoding='utf-8').readlines()
        
        # Split every line into pairs and normalize
        self.pairs = [[src_lines[i].strip(), tgt_lines[i].strip()] for i in range(len(src_lines))]
    
        # Reverse pairs
        if reverse: self.pairs = [list(reversed(p)) for p in pairs]
    
        print("Success!")
    
    def _filterPairs(self, max_cut):
        self.pairs = [[pair[0].split(' '), pair[1].split(' ')] for pair in self.pairs]
        if max_cut:
            self.pairs = [[pair[0][:self.max_length], pair[1][:self.max_length]] for pair in self.pairs]
        else:
            self.pairs = [pair for pair in self.pairs if self._filterPair(pair)]
        
    def _filterPair(self, p):
        return len(p[0]) <= self.max_length and len(p[1]) <= self.max_length and len(p[0]) > 1 and len(p[1]) > 1
    
    def _prepareVocab(self):
        for pair in self.pairs:
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
        tgt_lengths.append(item[3]+1)  # For sos or eos token
    return [src_indices, tgt_indices, src_lengths, tgt_lengths]