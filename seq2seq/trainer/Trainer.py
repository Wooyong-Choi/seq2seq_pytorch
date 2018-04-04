import os
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torch.nn.utils.rnn import pad_sequence   pytorch 0.3.0 or later

from seq2seq.dataset import sorted_collate_fn

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Trainer(object):
    """
    A trainer class supports our seq2seq model to train it easily
    """
    
    def __init__(self, model, dataset, gpu_id=-1, print_interval=1, plot_interval=1, checkpoint_interval=10, expr_path='experiment/'):
        super(Trainer, self).__init__()
        self.model = model
        self.dataset = dataset
        self.data_loader = None
        
        self.gpu_id = gpu_id
        
        self.print_interval = print_interval
        self.plot_interval = plot_interval
        self.checkpoint_interval = checkpoint_interval
        
        self.expr_path = expr_path
        if not os.path.exists(self.expr_path):
            os.makedirs(self.expr_path)
        
    def train(self, num_epoch, batch_size, optimizer=None, criterion=None):
        start = time.time()
        
        self.data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            collate_fn=sorted_collate_fn,
            num_workers=16
        )
        
        if optimizer == None:
            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        if criterion == None:
            criterion = nn.NLLLoss(size_average=True)
        
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
    
        for epoch in range(1, num_epoch + 1):
            for src_batch, tgt_batch, encoder_input_length in self.data_loader:
                optimizer.zero_grad()
                
                # prepare batch data
                encoder_input = self.prepareBatch(src_batch)
                decoder_input = self.prepareBatch(tgt_batch, appendSOS=True)
                decoder_target = self.prepareBatch(tgt_batch, appendEOS=True)
                if self.gpu_id != -1:
                    encoder_input = encoder_input.cuda(self.gpu_id)
                    decoder_input = decoder_input.cuda(self.gpu_id)
                    decoder_target = decoder_target.cuda(self.gpu_id)
        
                # forward model
                decoder_outputs = self.model(encoder_input, decoder_input, encoder_input_length)
            
                # calculate loss and back-propagate
                loss = criterion(decoder_outputs.contiguous().view(-1, self.model.output_size), decoder_target.contiguous().view(-1))
                loss.backward()
    
                optimizer.step()

                print_loss_total += loss.data[0]
                plot_loss_total += loss.data[0]
    
            if epoch % self.print_interval == 0:
                print_loss_avg = print_loss_total / self.print_interval
                print_loss_total = 0
                print('epoch:%3d (%3d%%) time:%s loss:%.4f' % (epoch, epoch/num_epoch*100, self._timeSince(start, epoch/num_epoch), print_loss_avg))
                
            if epoch % self.plot_interval == 0:
                plot_loss_avg = plot_loss_total / self.plot_interval
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
                
            if epoch % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
            
        self._showPlot(plot_losses)
        
        
    #TODO: 밑에 애들 utils 로 옮길까
    def prepareBatch(self, batch, appendSOS=False, appendEOS=False):
        SOS_IDX = self.data_loader.dataset.src_vocab.sos_idx
        EOS_IDX = self.data_loader.dataset.src_vocab.eos_idx
        
        batch_list = []
        for indices in batch:
            if appendSOS:
                batch_list.append(Variable(torch.LongTensor([SOS_IDX]+indices)))
            elif appendEOS:
                batch_list.append(Variable(torch.LongTensor(indices+[EOS_IDX])))
            else:
                batch_list.append(Variable(torch.LongTensor(indices)))
        batch_list = sorted(batch_list, key=lambda x: x.size(0), reverse=True)
        return self._pad_sequence(batch_list, batch_first=True)
    
    # from pytorch 0.3.0
    def _pad_sequence(self, sequences, batch_first=False, padding_value=0):
        """
        Pad a list of variable length Variables with zero
        """
        # assuming trailing dimensions and type of all the Variables
        # in sequences are same and fetching those from sequences[0]
        max_size = sequences[0].size()
        max_len, trailing_dims = max_size[0], max_size[1:]
        prev_l = max_len
        if batch_first:
            out_dims = (len(sequences), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(sequences)) + trailing_dims
        
        out_variable = Variable(sequences[0].data.new(*out_dims).fill_(padding_value))
        for i, variable in enumerate(sequences):
            length = variable.size(0)
            # temporary sort check, can be removed when we handle sorting internally
            if prev_l < length:
                raise ValueError("lengths array has to be sorted in decreasing order")
            prev_l = length
            # use index notation to prevent duplicate references to the variable
            if batch_first:
                out_variable[i, :length, ...] = variable
            else:
                out_variable[:length, i, ...] = variable
        
        return out_variable
    
    def _save_checkpoint(self, epoch):
        checkpoint_path = self.expr_path+self.model.name+str(epoch)+'.model'
        torch.save(self.model.state_dict(), checkpoint_path)
        
    def _showPlot(self, points):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)
        plt.savefig(self.expr_path+self.model.name+"_train_loss.png")
    
    def _asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
    def _timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self._asMinutes(s), self._asMinutes(rs))