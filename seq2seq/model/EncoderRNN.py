import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=self.bidirectional)

    def forward(self, input_seqs, input_lens, hidden):
        """
        Inputs is batch of sentences: BATCH_SIZE x MAX_LENGTH+1
        """
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lens, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden