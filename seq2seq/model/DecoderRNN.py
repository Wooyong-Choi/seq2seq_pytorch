import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .Attn import Attn

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, max_length, bidirection, gpu_id=-1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.gpu_id = gpu_id
        self.bidirectional_encoder = bidirection

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.attn = Attn('dot', hidden_size, self.gpu_id)
        self.out = nn.Linear(hidden_size, output_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.softmax = nn.LogSoftmax(dim=1)
            
    def forward(self, input_seqs, hidden, encoder_outputs):
        cur_batch_size, max_len = input_seqs.size()
        
        hidden = self._cat_directions(hidden) if self.bidirectional_encoder else hidden

        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(input_seqs)
        
        # Get current hidden state from input word and last hidden state
        outputs, hidden = self.rnn(embedded, hidden)
        
        # Calculate attention
        outputs, p_attn = self.attn(outputs, encoder_outputs)

        outputs = self.softmax(self.out(outputs))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return outputs, hidden, p_attn
    
    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h