# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import torch
import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import matplotlib.font_manager as fm

from seq2seq.model import Seq2seqModel

class Evaluator(object):
    """
    For Evaluation
    """
    def __init__(self, dataset, model, gpu_id=-1):
        super(Evaluator, self).__init__()
        self.dataset = dataset
        self.model = model
        self.gpu_id = gpu_id
        if self.gpu_id != -1:
            self.model.cuda(self.gpu_id)
        
    def evalModel(self, num, beam_size=-1, showAttn=False):
        pairs = []
        for i in range(num):
            test_pair = random.choice(self.dataset.test_pairs)
            pair, attn_weights = self.generateResponse(test_pair[0], beam_size=beam_size)
            pairs.append(pair)
            if showAttn == True:
                self.showAttention(i, pair[0], pair[1], attn_weights)
        return pairs
    
    def generateResponse(self, input, beam_size=-1):
        """
        Params:
        -------
        inputs: string of sentence
        beam_size: beam size for beam search
        """
        if (isinstance(input, str)):
            input_tokens = self.dataset._normalizeString(input)
            input_indices = self.dataset.src_vocab.sentence_to_indices(input_tokens)
            
        elif (isinstance(input, list) and isinstance(input[0], str)):
            input_tokens = input
            input_indices = self.dataset.src_vocab.sentence_to_indices(input)
            
        elif (isinstance(input, list) and isinstance(input[0], int)):
            input_tokens = self.dataset.vocab.indices_to_sentence(input)
            input_indices = input
            
        gen_sentences, attn_weights = self.model.sampleResponce(input_indices, self.dataset.src_vocab, self.dataset.tgt_vocab, beam_size=beam_size)
        
        return (input_tokens, gen_sentences), attn_weights
    
    def showAttention(self, idx, input_sentence, output_words, attentions):
        path = '/usr/share/fonts/truetype/MS/malgun.ttf'
        fontprop = fm.FontProperties(fname=path, size='medium')
        
        # Set up figure with colorbar
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(''.join(input_sentence)+' -- '+''.join(output_words), fontproperties=fontprop)
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)
    
        # Set up axes
        ax.set_xticklabels([''] + input_sentence + ['<eos>'], rotation=90, fontproperties=fontprop)
        ax.set_yticklabels([''] + output_words, fontproperties=fontprop)
    
        # Show label at every tick
        ax.xaxis.set_major_locator(tk.MultipleLocator(1))
        ax.yaxis.set_major_locator(tk.MultipleLocator(1))
    
        #plt.show()
        plt.savefig('{}.png'.format(idx))
    
    # TODO: utils로 빼자
    def loadModel(self, model_state_dict_path):
        self.model.load_state_dict(torch.load(model_state_dict_path))
        if self.gpu_id != -1:
            self.model.cuda(self.gpu_id)
        