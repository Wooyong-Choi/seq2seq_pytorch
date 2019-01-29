# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import torch
import random

from rouge import Rouge

from seq2seq.model import Seq2seqModel

class Evaluator(object):
    """
    For Evaluation
    """
    def __init__(self, dataset, model, device):
        super(Evaluator, self).__init__()
        self.dataset = dataset
        self.model = model
        self.device = device
        self.model.to(device)
        
    def evalModel(self, num, beam_size=-1, rand=True, rouge=False, showAttn=False):
        pairs = []
        attn_list = []
        for i in range(num):
            if rand:
                test_pair = random.choice(self.dataset.test_pairs)
            else:
                test_pair = self.dataset.test_pairs[i]
            gen_sentence, attn_weights = self.generateResponse(test_pair[0], beam_size=beam_size)
            pairs.append((*test_pair, gen_sentence))
            attn_list.append(attn_weights)
            if showAttn:
                self.showAttention(i, pair[0], pair[1], attn_weights)
            
        if rouge:
            rouge = Rouge()
            hyps, refs = zip(*[[' '.join(d[1]), ' '.join(d[2])] for d in pairs])
            scores = rouge.get_scores(hyps, refs, avg=True)
            return pairs, attn_list, scores
        else:
            return pairs, attn_list
        
    
    def generateResponse(self, input, beam_size=-1):
        """
        Params:
        -------
        inputs: string of sentence
        beam_size: beam size for beam search
        """
        if (isinstance(input, str)):
            input_tokens = self.dataset._normalizeString(input).split()
            input_indices = self.dataset.src_vocab.sentence2indice(input_tokens)
            
        elif (isinstance(input, list) and isinstance(input[0], str)):
            input_tokens = input
            input_indices = self.dataset.src_vocab.sentence2indice(input)
            
        elif (isinstance(input, list) and isinstance(input[0], int)):
            input_tokens = self.dataset.src_vocab.indice2sentence(input)
            input_indices = input
            
        gen_sentences, attn_weights = self.model.sampleResponce(
            input_indices, self.dataset.src_vocab, self.dataset.tgt_vocab, beam_size=beam_size
        )
        
        return gen_sentences, attn_weights
    
    
    # TODO: utils로 빼자
    def loadModel(self, model_state_dict_path):
        self.model.load_state_dict(torch.load(model_state_dict_path))
        self.model.to(self.device)
        