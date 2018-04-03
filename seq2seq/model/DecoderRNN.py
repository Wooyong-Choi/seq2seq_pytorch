import torch
import torch.nn as nn
from torch.autograd import Variable

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input_seqs, hidden):
        """
        Inputs is batch of sentences: BATCH_SIZE x MAX_LENGTH+1
        """
        embedded = self.embedding(input_seqs)
        outputs, hidden = self.gru(embedded, hidden)
        outputs = self.out(outputs)
        outputs = self.softmax(outputs.transpose(0, 2))  # do not need to transpose tensor in pytorch 0.4.0, just do softmax(dim=2)
        outputs = outputs.transpose(0, 2)
        return outputs, hidden