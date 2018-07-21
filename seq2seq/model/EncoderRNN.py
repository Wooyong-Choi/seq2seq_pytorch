import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, n_layers, bidirectional, encode_mode, gpu_id):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.encode_mode = encode_mode
        self.gpu_id = gpu_id

        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=0)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True, bidirectional=self.bidirectional)

    def forward(self, input_seqs, input_lens, hidden, layout):
        """
        Inputs is batch of sentences: BATCH_SIZE x MAX_LENGTH+1
        """
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lens, batch_first=True)
        outputs, hidden = self.rnn(packed) # default zero hidden
        outputs, output_lengths = pad_packed_sequence(outputs, batch_first=True)
        
        if self.encode_mode == 'max':
            outputs = self.getMaxedContext(outputs, layout, self.input_size, self.hidden_size*2 if self.bidirectional else self.hidden_size)
        if self.encode_mode == 'avg':
            outputs = self.getAvgedContext(outputs, layout, self.input_size, self.hidden_size*2 if self.bidirectional else self.hidden_size)
            
        return outputs, hidden
    
    def getMaxedContext(self, context, layout, input_size, hidden_size):
        max_word_lens = max([len(l)-1 for l in layout])

        maxed_contexts = []
        for i, l in enumerate(layout):
            cur_batch_context = []
            for j in range(len(l)-1):
                prev = l[j]
                next = l[j+1]
                num = next - prev
                if num != 0:
                    cur_batch_context.append(torch.cat([context[i, prev:next].max(dim=0)[0]], dim=0))
                    
            maxed_context = torch.stack(cur_batch_context, dim=0)
            padding_size = max_word_lens - len(cur_batch_context)
            if padding_size != 0:
                padding_vec = torch.zeros((padding_size, hidden_size)).cuda(self.gpu_id)
                maxed_context = torch.cat((maxed_context, padding_vec), dim=0)
            maxed_contexts.append(maxed_context)

        maxed_contexts = torch.stack(maxed_contexts, dim=0)
        return maxed_contexts
    
    def getAvgedContext(self, context, layout, input_size, hidden_size):
        max_word_lens = max([len(l)-1 for l in layout])
    
        avged_contexts = []
        for i, l in enumerate(layout):
            cur_batch_context = []
            for j in range(len(l)-1):
                prev = l[j]
                next = l[j+1]
                num = next - prev
                if num != 0:
                    cur_batch_context.append(torch.cat([context[i, prev:next].sum(dim=0) / num], dim=0))
                    
            avged_context = torch.stack(cur_batch_context, dim=0)
            padding_size = max_word_lens - len(cur_batch_context)
            if padding_size != 0:
                padding_vec = torch.zeros((padding_size, hidden_size)).cuda(self.gpu_id)
                avged_context = torch.cat((avged_context, padding_vec), dim=0)
            avged_contexts.append(avged_context)
    
        avged_contexts = torch.stack(avged_contexts, dim=0)
        return avged_contexts