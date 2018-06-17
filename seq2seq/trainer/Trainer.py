import os
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from seq2seq.dataset import sorted_collate_fn
from seq2seq.utils import masked_cross_entropy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Trainer(object):
    """
    A trainer class supports our seq2seq model to train it easily
    """
    
    def __init__(self, model, dataset, gpu_id=-1, print_interval=1, plot_interval=1, checkpoint_interval=10, eval_interval=10, expr_path='experiment/'):
        super(Trainer, self).__init__()
        self.model = model
        self.dataset = dataset
        self.data_loader = None
        
        self.gpu_id = gpu_id
        
        self.print_interval = print_interval
        self.plot_interval = plot_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        
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
            criterion = nn.NLLLoss(size_average=True, ignore_index=0)
        
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
    
        for epoch in range(1, num_epoch + 1):
            for src_batch, tgt_batch, src_length, tgt_length in self.data_loader:
                optimizer.zero_grad()
                
                # prepare batch data
                src_batch = self.prepareBatch(src_batch)
                tgt_batch = self.prepareBatch(tgt_batch, appendSOS=True, appendEOS=True)
                if self.gpu_id != -1:
                    src_batch = src_batch.cuda(self.gpu_id)
                    tgt_batch = tgt_batch.cuda(self.gpu_id)
                
                # forward model
                decoder_outputs = self.model(src_batch, tgt_batch[:,:-1], src_length)
                
                start_time = time.time()
            
                # calculate loss and back-propagate
                # tgt_batch[:,1:] : remove SOS tokens in all mini-batch
                loss = criterion(decoder_outputs.view(-1, self.model.output_size), tgt_batch[:,1:].contiguous().view(-1))
                loss.backward()
    
                optimizer.step()
        
                print_loss_total += loss.item()
                plot_loss_total += loss.item()
    
            if epoch % self.print_interval == 0:
                print_loss_avg = print_loss_total / self.print_interval
                print('epoch:%3d (%3d%%) time:%20s loss:%.4f' % (epoch, epoch/num_epoch*100, self._timeSince(start, epoch/num_epoch), print_loss_avg))
                print_loss_total = 0
                
            if epoch % self.plot_interval == 0:
                plot_loss_avg = plot_loss_total / self.plot_interval
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
                
            if epoch % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
                
            #if epoch % self.eval_interval == 0:
                # eval
        
        if self.plot_interval != -1:
            self._showPlot(plot_losses)
        
        
    #TODO: 밑에 애들 utils 로 옮길까
    def prepareBatch(self, batch, appendSOS=False, appendEOS=False):
        SOS_IDX = self.data_loader.dataset.src_vocab.sos_idx
        EOS_IDX = self.data_loader.dataset.src_vocab.eos_idx
        PAD_IDX = self.data_loader.dataset.src_vocab.pad_idx
        
        batch_list = []
        for indices in batch:
            pad_num = self.dataset.max_length - len(indices)
            if appendSOS:
                batch_list.append(torch.LongTensor([SOS_IDX]+indices+([PAD_IDX]*pad_num)))
            elif appendEOS:
                batch_list.append(torch.LongTensor(indices+[EOS_IDX]+([PAD_IDX]*pad_num)))
            else:
                batch_list.append(torch.LongTensor(indices+([PAD_IDX]*(pad_num))))
        return Variable(torch.stack(batch_list, dim=0))
    
    def _get_eval_loss(self):
        for src_batch, tgt_batch, src_length, tgt_length in self.data_loader:
            optimizer.zero_grad()
            
            # prepare batch data
            src_batch = self.prepareBatch(src_batch)
            tgt_batch_sos = self.prepareBatch(tgt_batch, appendSOS=True)
            tgt_batch_eos = self.prepareBatch(tgt_batch, appendEOS=True)
            if self.gpu_id != -1:
                src_batch = src_batch.cuda(self.gpu_id)
                tgt_batch_sos = tgt_batch_sos.cuda(self.gpu_id)
                tgt_batch_eos = tgt_batch_eos.cuda(self.gpu_id)
            
            # forward model
            decoder_outputs = self.model(src_batch, tgt_batch_sos, src_length)
        
            # calculate loss and back-propagate
            start_time = time.time()
            #print(decoder_outputs.size(), tgt_batch_eos.size())
            #loss = masked_cross_entropy(decoder_outputs.contiguous(), tgt_batch_eos.contiguous(), tgt_length, self.gpu_id)
            loss = criterion(decoder_outputs.view(-1, self.model.output_size), tgt_batch_eos.view(-1))
            loss.backward()

            optimizer.step()
    
            print_loss_total += loss.item()
            plot_loss_total += loss.item()
    
    def _save_checkpoint(self, epoch):
        checkpoint_path = self.expr_path+self.model.name+str(epoch)+'.model'
        torch.save(self.model.state_dict(), checkpoint_path)
        
    def _showPlot(self, points):
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