import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from seq2seq.model.attn import Attn

class DecoderRNN(nn.Module):
    
    def __init__(self, n_layers, hidden_size, emb_size, output_size, dropout_p):
        
        super(DecoderRNN, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=0)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True, dropout=dropout_p)
        self.attn = Attn(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        batch_size = decoder_input.size(0)
        output_size = decoder_input.size(1)
        
        embedded = self.embedding(decoder_input)
        decoder_output, decoder_hidden = self.rnn(embedded, decoder_hidden)

        decoder_output, attn_weight = self.attn(decoder_output, encoder_outputs)

        decoder_output = self.out(decoder_output.contiguous().view(-1, self.hidden_size))
        decoder_output = self.softmax(decoder_output).view(batch_size, output_size, -1)
        
        return decoder_output, decoder_hidden, attn_weight
