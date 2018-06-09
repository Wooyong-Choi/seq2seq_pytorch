import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .Attn import Attn

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, max_length, bidirection, use_attention, gpu_id=-1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.gpu_id = gpu_id
        self.bidirectional_encoder = bidirection
        self.use_attention = use_attention

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)      
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        if self.use_attention:
            self.attn = Attn('dot', hidden_size, self.gpu_id)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
            
    def forward(self, input_seqs, hidden, encoder_outputs):
        hidden = self._cat_directions(hidden) if self.bidirectional_encoder else hidden
        
        cur_batch_size, max_len = input_seqs.size()
        
        outputs = Variable(torch.zeros(max_len, cur_batch_size, self.output_size))
        if self.gpu_id != -1:
            outputs = outputs.cuda(self.gpu_id)

        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(input_seqs)
        
        attn_weights = None
        if self.use_attention:
            outputs, attn_weights = self.attn(embedded, encoder_outputs)
        
        # Get current hidden state from input word and last hidden state
        outputs, hidden = self.gru(outputs, hidden)

        # Finally predict next token (Luong eq. 6, without softmax)
        outputs = self.out(outputs.contiguous().view(-1, self.hidden_size))
        outputs = self.softmax(outputs).view(cur_batch_size, max_len, -1)
        
        # Return final output, hidden state, and attention weights (for visualization)
        return outputs, hidden, attn_weights
    
    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder and h.size()[0] == 2:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h