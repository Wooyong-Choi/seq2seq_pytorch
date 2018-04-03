import operator

class Vocab:
    """
    A vocabulary class has word dictionary.
    """
    
    def __init__(self):
        self.num_symbol = 4
        self.pad_tok, self.sos_tok, self.eos_tok, self.unk_tok = ('<PAD>', '<SOS>', '<EOS>', '<UNK>')        
        self.pad_idx, self.sos_idx , self.eos_idx, self.unk_idx = (0, 1, 2, 3)
        
        self.word2count = {}
        self.word2index = {}
        self.index2word = dict(
            list(zip([self.pad_idx, self.sos_idx, self.eos_idx, self.unk_idx],
                     [self.pad_tok, self.sos_tok, self.eos_tok, self.unk_tok]))
        )
        self.n_words = len(self.index2word)

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1            
        
    def makeVocabDict(self, vocab_size):
        """
        Make index2word and word2index with added sentences and words before.
        The indices of word is decided by its frequency.
        """
        # make vocab dictionary with min(given vocab size, counted vocab size)
        vocab_size = min(vocab_size, len(self.word2count)+4)
        
        # sort vocab dictionary using frequency
        sorted_vocab = sorted(self.word2count.items(), key=operator.itemgetter(1), reverse=True)[:vocab_size]
        
        # update index2word dictionary
        sorted_i2w = {i+self.num_symbol:sorted_vocab[i][0] for i in range(vocab_size-self.num_symbol)}
        self.index2word.update(sorted_i2w)
        
        # update word2index dictionary
        self.word2index = {v:k for k, v in self.index2word.items()}
        
        # update a num of words
        self.n_words = vocab_size
        
    def sequence_to_indices(self, sentence):
        return [self.word2index[word] if word in self.word2index else self.unk_idx for word in sentence]
    
    def indices_to_sentence(self, indices):
        return [self.index2word[idx] if idx in self.index2word else self.unk_tok for idx in indices]