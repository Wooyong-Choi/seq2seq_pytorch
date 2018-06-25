import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .Attn import Attn

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, emb_size, output_size, n_layers, max_length, bidirection, gpu_id=-1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.gpu_id = gpu_id
        self.bi_encoder = bidirection

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=0)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.attn = Attn(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward_step(self, input_var, hidden, encoder_outputs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        
        embedded = self.embedding(input_var)
#        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        output, attn_weight = self.attn(output, encoder_outputs)

        output = self.out(output.contiguous().view(-1, self.hidden_size))
        output = self.softmax(output).view(batch_size, output_size, -1)
        return output, hidden, attn_weight

    def forward(self, decoder_input, encoder_hidden, encoder_outputs):
        decoder_hidden = self._cat_directions(encoder_hidden) if self.bi_encoder else encoder_hidden
        
        decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
        
        return decoder_output, decoder_hidden, attn
    
    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h