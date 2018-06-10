import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, method, hidden_size, gpu_id=-1):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.gpu_id = gpu_id
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        # one step input
        if hidden.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False
            
        cur_batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_weights = self.score(hidden, encoder_outputs)  # align
        if self.gpu_id != -1:
            attn_weights = attn_weights.cuda(self.gpu_id)

        attn_weights = F.softmax(attn_weights, dim=1)

        context = torch.bmm(attn_weights, encoder_outputs)
        
        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        concat_input = torch.cat((context, hidden), 2)
        concat_output = self.concat(concat_input)  # attn_h
        if self.method in ["dot", "general"]:
            concat_output = F.tanh(concat_output)
            
        if one_step:
            concat_output = concat_output.squeeze(1)
            attn_weight = attn_weight.squeeze(1)
        
        return concat_output.contiguous(), attn_weights.contiguous()
    
    def score(self, h_t, h_s):
        if self.method == 'dot':
            h_s = h_s.transpose(1, 2)   # (batch, d, s_len) --> (batch, s_len, d)
            energies = torch.bmm(h_t, h_s)  # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return energies
        
        elif self.method == 'general':
            h_s = self.attn(h_s)
            h_s = h_s.transpose(1, 2)   # (batch, d, s_len) --> (batch, s_len, d)
            energies = torch.bmm(h_t, h_s)
            return energies
        
        elif self.method == 'concat':
            concated = self.attn(torch.cat((h_t, h_s), 1))
            concated = F.tanh(concated)
            concated = concated.transpose(1, 2)   # (batch, d, s_len+t_len) --> (batch, s_len+t_len, d)
            energies = torch.bmm(self.v, energy)
            return energies