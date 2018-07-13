import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attn(nn.Module):
    def __init__(self, dim, attn_mode, gpu_id):
        super(Attn, self).__init__()
        self.attn_mode = attn_mode
        self.gpu_id = gpu_id
        self.W = nn.Linear(dim*2, dim)

    def forward(self, output, context, layout):
        batch_size = output.size(0)
        output_size = output.size(1)
        hidden_size = output.size(2)
        input_size = context.size(1)
        
        # h_t = output
        # h_s = context
        # dot score
        # (batch, out_len, dim) * (batch, dim, in_len) -> (batch, out_len, in_len)
        score = torch.bmm(output, context.transpose(1, 2))
        if self.attn_mode == 'max':
            score = self.getMaxedScore(score, layout, input_size, output_size)
        elif self.attn_mode == 'avg':
            score = self.getAvgedScore(score, layout, input_size, output_size)
        align = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # c_t = derived context
        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        aligned_context = torch.bmm(align, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((aligned_context, output), dim=2)
        
        # output -> (batch, out_len, dim)
        output = F.tanh(self.W(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, align
    
    def getAvgedScore(self, score, layout, input_size, output_size):
        avged_scores = []
        for i, l in enumerate(layout):
            cur_batch_score = []
            for j in range(len(l)-1):
                prev = l[j]
                next = l[j+1]
                num = next - prev
                # 띄어쓰기 중첩이 아니면 평균 계산 후 평균으로 매꿈
                if num != 0:
                    cur_batch_score.append(torch.stack([score[i, :, prev:next].sum(dim=1) / num] * num, dim=1))
            avged_score = torch.cat(cur_batch_score, dim=1)
            padded_size = input_size-avged_score.size()[1]
            if padded_size != 0:
                avged_score = torch.cat((avged_score, torch.zeros(output_size, padded_size).cuda(self.gpu_id)), dim=1)
            
            avged_scores.append(avged_score)
        
        avged_scores = torch.stack(avged_scores, dim=0)
        return avged_scores
    
    def getMaxedScore(self, score, layout, input_size, output_size):
        maxed_scores = []
        for i, l in enumerate(layout):
            cur_batch_score = []
            for j in range(len(l)-1):
                prev = l[j]
                next = l[j+1]
                num = next - prev
                # 띄어쓰기 중첩이 아니면 평균 계산 후 평균으로 매꿈
                if num != 0:
                    cur_batch_score.append(torch.stack([score[i, :, prev:next].max(dim=1)[0]] * num, dim=1))
            maxed_score = torch.cat(cur_batch_score, dim=1)
            padded_size = input_size-maxed_score.size()[1]
            if padded_size != 0:
                maxed_score = torch.cat((maxed_score, torch.zeros(output_size, padded_size).cuda()), dim=1)
            
            maxed_scores.append(maxed_score)
        
        maxed_scores = torch.stack(maxed_scores, dim=0)
        return maxed_scores

'''
class Attn(nn.Module):
    def __init__(self, method, hidden_size, gpu_id=-1):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.gpu_id = gpu_id
        
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, hidden, encoder_outputs):
        """
        Returns:
          (`FloatTensor`, `FloatTensor`):
          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """
        
        batch, source, dim = encoder_outputs.size()
        batch, target, dim = hidden.size()
        
        align = self._score(hidden, encoder_outputs)
        
        align_vectors = self.softmax(align.view(batch * target, source))
        align_vectors = align_vectors.view(batch, target, source)
                
        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, encoder_outputs)
                
        # concatenate
        concat_c = torch.cat([c, hidden], 2).view(batch * target, dim * 2)
        attn_h = self.fc(concat_c).view(batch, target, dim)
        attn_h = self.tanh(attn_h)
        
        # one step
        attn_h = attn_h.squeeze(1)
        align_vectors = align_vectors.squeeze(1)
            
        return attn_h, align_vectors
    
    def _score(self, h_t, h_s):
        h_s = h_s.transpose(1, 2)
        return torch.bmm(h_t, h_s)
    
class Attn(nn.Module):
    def __init__(self, method, hidden_size, gpu_id=-1):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.gpu_id = gpu_id

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size()[1]

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if self.gpu_id == -1: attn_energies = attn_energies.cuda(GPU_ID)

        # Calculate energies for each encoder output
        print(hidden.size(), encoder_outputs.size())
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy
        
'''
