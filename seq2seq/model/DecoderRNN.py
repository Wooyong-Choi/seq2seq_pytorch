import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .Attn import Attn

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, max_length, dropout_p=0.1, gpu_id=-1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.gpu_id = gpu_id

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)      
        self.dropout = nn.Dropout(dropout_p)        
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attn = Attn('dot', hidden_size, self.gpu_id)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
            
    def forward(self, input_seqs, hidden, encoder_outputs):
        cur_batch_size, max_len = input_seqs.size()
        
        outputs = Variable(torch.zeros(max_len, cur_batch_size, self.output_size))
        if self.gpu_id != -1:
            outputs = outputs.cuda(self.gpu_id)

        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout(embedded)

        # Get current hidden state from input word and last hidden state
        rnn_outputs, hidden = self.gru(embedded, hidden)
        
        outputs, attn_weights = self.attn(rnn_outputs, encoder_outputs)

        # Finally predict next token (Luong eq. 6, without softmax)
        outputs = self.out(outputs)

        outputs = self.softmax(outputs.transpose(0, 2))
        outputs = outputs.transpose(0, 2)
        
        # Return final output, hidden state, and attention weights (for visualization)
        return outputs, hidden, attn_weights