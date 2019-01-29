import os
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from seq2seq.data import sorted_collate_fn
from seq2seq.data import PAD_IDX, SOS_IDX, EOS_IDX

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Trainer(object):
    """
    A trainer class supports our seq2seq model to train it easily
    """
    
    def __init__(self, model, dataset, device, print_interval=1, plot_interval=1, checkpoint_interval=10, eval_interval=10, expr_path='experiment/'):
        super(Trainer, self).__init__()
        self.model = model
        self.dataset = dataset
        
        self.device = device
        
        self.print_interval = print_interval
        self.plot_interval = plot_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        
        self.expr_path = expr_path
        if not os.path.exists(self.expr_path):
            os.makedirs(self.expr_path)
        
    def train(self, num_epoch, batch_size, lr_val=1e-3, start_decay=0, lr_decay=1, optimizer=None, criterion=None):
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
            criterion = nn.NLLLoss(reduction='mean', ignore_index=PAD_IDX).to(self.device)
        
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        
        if os.path.exists(self.expr_path+'log.txt'):
            os.remove(self.expr_path+'log.txt')
    
        for epoch in range(1, num_epoch+1):
            for src_batch, tgt_batch, src_length, tgt_length in self.data_loader:
                optimizer.zero_grad()
                
                # prepare batch data
                enc_input = pad_sequence(src_batch, batch_first=True).to(self.device)
                dec_input = pad_sequence(tgt_batch, batch_first=True).to(self.device)
                
                # forward model
                decoder_outputs = self.model(enc_input, dec_input, src_length)
                
                start_time = time.time()
            
                # calculate loss and back-propagate
                loss = criterion(decoder_outputs[:,:-1].contiguous().view(-1, self.model.output_size),
                                 dec_input[:,1:].contiguous().view(-1))
                loss.backward()
    
                optimizer.step()
        
                print_loss_total += loss.item()
                plot_loss_total += loss.item()

            # decay learning rate
            if start_decay != 0:
                self._lr_scheduler(optimizer, lr_val, epoch, start_decay=start_decay, decay_factor=lr_decay)
                                   
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
    
    # TODO:
    #def _get_eval_loss(self):
        
    
    def _save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.expr_path, 'ep'+str(epoch)+'.model')
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def _lr_scheduler(self, optimizer, init_lr, iter, start_decay, decay_factor=0.9):
        """Decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param start_decay how frequently decay occurs, default is 1
            :param decay_factor is a decay factor
        """
        lr = init_lr*decay_factor**(iter-start_decay+1 if iter-start_decay >= -1 else 0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
        return lr
    
    def _showPlot(self, points):
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)
        plt.savefig(os.path.join(self.expr_path, "train_loss.png"))
    
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