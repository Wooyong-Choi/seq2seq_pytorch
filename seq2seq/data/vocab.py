import operator
from collections import Counter


PAD_TOK, SOS_TOK, EOS_TOK, UNK_TOK = ('<PAD>', '<SOS>', '<EOS>', '<UNK>')        
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = (0, 1, 2, 3)


class Vocab:
    """
    A vocabulary class has word counter and dictionary.
    """
    
    def __init__(self):
        self.num_symbol = 4
        
        self.word2count = Counter()
        self.word2index = {}
        self.index2word = {PAD_IDX:PAD_TOK,
                           SOS_IDX:SOS_TOK,
                           EOS_IDX:EOS_TOK,
                           UNK_IDX:UNK_TOK}
        self.n_words = len(self.index2word)

    def addSentence(self, sentence):
        """
        Add a sentence to word counter
        """
        self.word2count.update(sentence)
        self.n_words = len(self.word2count) + 4
        
    def makeVocabDict(self, vocab_size):
        """
        Make index2word and word2index with added sentences and words before.
        The indices of word is decided by its frequency.
        """
        # make vocab dictionary with min(given vocab size, counted vocab size)
        vocab_size = min(vocab_size, len(self.word2count)+4)
        
        # sort vocab dictionary using frequency
        sorted_vocab = self.word2count.most_common()
        
        # update index2word dictionary
        sorted_i2w = {i+self.num_symbol:sorted_vocab[i][0] for i in range(vocab_size-self.num_symbol)}
        self.index2word.update(sorted_i2w)
        
        # update word2index dictionary
        self.word2index = {v:k for k, v in self.index2word.items()}
        
        # update a num of words
        self.n_words = vocab_size
        
    def sentence2indice(self, sentence):
        return [self.word2index[word] if word in self.word2index else UNK_IDX for word in sentence]
    
    def indice2sentence(self, indice):
        return [self.index2word[idx] if idx in self.index2word else UNK_TOK for idx in indice]