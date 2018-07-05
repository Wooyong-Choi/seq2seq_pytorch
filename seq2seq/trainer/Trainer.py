import os
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from seq2seq.dataset import sorted_collate_fn

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
        
    def train(self, num_epoch, batch_size, lr_val=1e-3, optimizer=None, criterion=None):
        start = time.time()
        
        print('Start to train')
        
        self.data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            collate_fn=sorted_collate_fn,
            num_workers=16
        )
        
        if optimizer == None:
            optimizer = optim.Adam(self.model.parameters(), lr=lr_val)
        if criterion == None:
            criterion = nn.NLLLoss(size_average=True, ignore_index=self.data_loader.dataset.src_vocab.pad_idx).cuda(self.gpu_id)
            #criterion = nn.NLLLoss(size_average=True, ignore_index=self.data_loader.dataset.src_vocab.pad_idx).cuda(self.gpu_id)
        
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        
        if os.path.exists(self.expr_path+'log.txt'):
            os.remove(self.expr_path+'log.txt')
    
        for epoch in range(1, num_epoch+1):
            for src_batch, tgt_batch, src_length, tgt_length, src_layout in self.data_loader:
                optimizer.zero_grad()
                
                # prepare batch data
                enc_input = self.prepareBatch(src_batch, max(src_length))
                dec_input = self.prepareBatch(tgt_batch, max(tgt_length), appendSOS=True)
                dec_target = self.prepareBatch(tgt_batch, max(tgt_length), appendEOS=True)
                if self.gpu_id != -1:
                    enc_input = enc_input.cuda(self.gpu_id)
                    dec_input = dec_input.cuda(self.gpu_id)
                    dec_target = dec_target.cuda(self.gpu_id)
                
                # forward model
                decoder_outputs = self.model(enc_input, dec_input, src_length, src_layout)
                
                start_time = time.time()
            
                # calculate loss and back-propagate
                loss = criterion(decoder_outputs.view(-1, self.model.output_size), dec_target.view(-1))
                loss.backward()
    
                optimizer.step()
        
                print_loss_total += loss.item()
                plot_loss_total += loss.item()
    
            if epoch % self.print_interval == 0:
                print_loss_avg = print_loss_total / self.print_interval
                log_str = 'epoch:%3d (%3d%%) time:%25s loss:%.4f' % (epoch, epoch/num_epoch*100, self._timeSince(start, epoch/num_epoch), print_loss_avg)
                print(log_str)
                print_loss_total = 0
                with open(self.expr_path+'log.txt', 'a') as fp:
                    fp.write(log_str + '\n')
                
            if epoch % self.plot_interval == 0:
                plot_loss_avg = plot_loss_total / self.plot_interval
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
                
            if epoch % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
                
            # TODO:
            #if epoch % self.eval_interval == 0:
                # eval
        
        if self.plot_interval != -1:
            self._showPlot(plot_losses)
        
        
    #TODO: 밑에 애들 utils 로 옮길까
    def prepareBatch(self, batch, max_len, appendSOS=False, appendEOS=False):
        SOS_IDX = self.data_loader.dataset.src_vocab.sos_idx
        EOS_IDX = self.data_loader.dataset.src_vocab.eos_idx
        PAD_IDX = self.data_loader.dataset.src_vocab.pad_idx
        
        batch_list = []
        for indices in batch:
            pad_num = max_len - len(indices)
            if appendSOS:
                batch_list.append(torch.LongTensor([SOS_IDX]+indices+([PAD_IDX]*pad_num)))
            elif appendEOS:
                batch_list.append(torch.LongTensor(indices+[EOS_IDX]+([PAD_IDX]*pad_num)))
            else:
                batch_list.append(torch.LongTensor(indices+([PAD_IDX]*(pad_num))))
        return Variable(torch.stack(batch_list, dim=0))
    
    # TODO:
    #def _get_eval_loss(self):
        
    
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
        return '%10s (- %10s)' % (self._asMinutes(s), self._asMinutes(rs))