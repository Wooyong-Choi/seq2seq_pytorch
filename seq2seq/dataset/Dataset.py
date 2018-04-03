import re
import sys
from multiprocessing.dummy import Pool as ThreadPool

from tqdm import tqdm
from torch.utils.data import Dataset

from .Vocab import Vocab

class Dataset(Dataset):
    def __init__(self, src_file_path, tgt_file_path, max_length, src_vocab_size=sys.maxsize, tgt_vocab_size=sys.maxsize):
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
        
    def __getitem__(self, index):
        src_indices = self.src_vocab.sequence_to_indices(self.train_pairs[index][0])
        tgt_indices = self.tgt_vocab.sequence_to_indices(self.train_pairs[index][1])
        src_lengths = len(self.train_pairs[index][0])
        return (src_indices, tgt_indices, src_lengths)
                
    def __len__(self):
        return len(self.train_pairs)
    
    def prepareData(self, tagger, reverse=False):
        self._readData()
        print("Read %s sentence pairs" % len(self.pairs))
        
        self._filterPairs(tagger)
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
        self.pairs = [[self._normalizeString(src_lines[i][:-1]), self._normalizeString(tgt_lines[i][:-1])] for i in range(len(src_lines))]
    
        # Reverse pairs
        if reverse: self.pairs = [list(reversed(p)) for p in pairs]
    
        print("Success!")
    
    def _normalizeString(self, s):
        s = re.sub('[^가-힝0-9a-zA-Z\\s]', '', s)
        return s
    
    def _filterPairs(self, tagger):
        self.pairs = [[tagger.morphs(pair[0]), tagger.morphs(pair[1])] for pair in self.pairs if self._filterPair(pair)]
        
    def _filterPair(self, p):
        return len(p[0]) <= self.max_length and len(p[1]) <= self.max_length
    
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
    '''
    Sort data using source sentences lengths in decreasing order
    for packing padded sequence
    '''
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    src_indices = [item[0] for item in batch]
    tgt_indices = [item[1] for item in batch]
    src_lengths = [item[2] for item in batch]
    return [src_indices, tgt_indices, src_lengths]