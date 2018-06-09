import torch
import random

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
        
    def evalModel(self, num, beam_size=-1):
        pairs = []
        for i in range(num):
            test_pair = random.choice(self.dataset.test_pairs)
            pairs.append(self.generateResponse(test_pair[0], beam_size=beam_size))
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
            
        return [input_tokens,
                self.model.sampleResponce(input_indices, self.dataset.src_vocab, self.dataset.tgt_vocab, beam_size=beam_size)]
    
    # TODO: utils로 빼자
    def loadModel(self, model_state_dict_path):
        self.model.load_state_dict(torch.load(model_state_dict_path))
        if self.gpu_id != -1:
            self.model.cuda(self.gpu_id)
        