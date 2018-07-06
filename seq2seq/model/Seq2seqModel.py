import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
from seq2seq.utils import Beam

class Seq2seqModel(nn.Module):
    """
    A seq2seq model has encoder and encoder using RNN.
    """
    
    def __init__(self, name, input_size, emb_size, hidden_size, output_size,
                 max_src_len, max_tgt_len,
                 dropout_p=0.1, bidirectional=False, use_attention=False, gpu_id=-1):
        super(Seq2seqModel, self).__init__()
        self.name = name
        self.input_size = input_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = 1
        
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        self.num_direction = 2 if bidirectional else 1
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.gpu_id = gpu_id
        
        """
        Encoder uses GRU with embedding layer and packing sequence
        """
        self.encoder = EncoderRNN(input_size, emb_size, hidden_size, self.n_layers, self.bidirectional)
        
        """
        Decoder uses GRU with embedding layer, fully connected layer and log-softmax
        """
        self.decoder = DecoderRNN(self.hidden_size*2 if self.bidirectional else self.hidden_size, emb_size, self.output_size,
                                  self.n_layers, self.dropout_p, self.max_tgt_len, self.bidirectional, self.gpu_id)

        if self.gpu_id != -1:
            self.cuda(self.gpu_id)
            
    def forward(self, src_batch, tgt_batch, src_batch_lengths, src_layout):
        """
        Params:
        -------
        src_batch: sequence of word indices,  "How are you? <PAD>"
        tgt_batch: sequence of word indices,  "<SOS> I'm fine. <PAD>"
        src_batch_lengths: lengths of source batch to use pack_padded_sequence
        tgt_batch_lengths: lengths of target batch to use masked cross entropy
        
        Returns:
        --------
        decoder_outputs : teacher forced outputs of decoder
        """
        cur_batch_size = src_batch.size()[0]  # batch_first
        
        # Encoder : sentence -> context
        encoder_hidden = self.initHidden(cur_batch_size)
        encoder_outputs, encoder_hidden = self.encoder(src_batch, src_batch_lengths, encoder_hidden)
        
        # Decoder : context -> response
        decoder_hidden = encoder_hidden
        decoder_context = encoder_outputs
        decoder_outputs, decoder_hidden, attn_weights = self.decoder(tgt_batch, decoder_hidden, decoder_context, src_layout)
        
        return decoder_outputs
    
    def initHidden(self, cur_batch_size):
        result = Variable(torch.zeros(1 * self.num_direction, cur_batch_size, self.hidden_size))
        if self.gpu_id != -1:
            return result.cuda(self.gpu_id)
        else:
            return result
    
    def sampleResponce(self, indices, layout, src_vocab, tgt_vocab, beam_size=-1):
        self.eval()
        
        encoder_input = Variable(torch.LongTensor(indices)).unsqueeze(0)
        encoder_input = encoder_input.cuda(self.gpu_id) if self.gpu_id != -1 else encoder_input
        encoder_input_length = [encoder_input.size()[1]]
        
        # Encoder
        encoder_hidden = self.initHidden(1)  # batch_size = 1
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_input_length, encoder_hidden)
    
        # Decoder
        decoder_hidden = encoder_hidden
        decoder_context = encoder_outputs
        
        decoded = []
        
        decoded_words = []
        attn_weights = []
        
        # top 1 decoder
        if beam_size == -1 or beam_size == 1:
            decoded = [tgt_vocab.sos_idx]
            for i in range(self.max_tgt_len+1):
                decoder_input = Variable(torch.LongTensor([decoded[-1]])).unsqueeze(0)
                decoder_input = decoder_input.cuda(self.gpu_id) if self.gpu_id != -1 else decoder_input
                
                decoder_output, decoder_hidden, attn_weight = self.decoder(decoder_input, decoder_hidden, decoder_context, layout)
                attn_weights.append(attn_weight.detach().squeeze(1).cpu())
                decoder_output = decoder_output.view(-1) # 1*1*10000 -> 10000
                
                # find candidates of candidates
                topVecs, topIdxs = decoder_output.topk(1)
                decoded.append(topIdxs.item())
                
                if topIdxs.item() == tgt_vocab.eos_idx:
                    break
            
            attn_weights = torch.cat(attn_weights, dim=0)
            
        # top k decoder with beam search
        else:
            elected_cand = self._decodeWithBeamSearch(decoder_hidden, decoder_context, beam_size, tgt_vocab.sos_idx, tgt_vocab.eos_idx, layout)
            decoded = elected_cand.seq
            attn_weights = elected_cand.attn
            
        # indices -> word sequence
        for idx in decoded[1:]:
            decoded_words.append(tgt_vocab.index2word[idx])
        
        self.train()
        return decoded_words, attn_weights
    
    def _decodeWithBeamSearch(self, decoder_hidden, decoder_context, beam_size, SOS_IDX, EOS_IDX, layout):
        """
        Find best sequence using beam search
        """
        beam = Beam(beam_size, decoder_hidden, SOS_IDX, EOS_IDX)

        # with beam search
        for i in range(self.max_tgt_len+1):
            # beam result can be searched beam board or final result of beam search
            self._beamSearchStep(beam, decoder_context, layout)
            # if first ranked sequence is ended with EOS_IDX, return it
            if beam.early_end is not None:
                return beam.early_end
        # no early end
        return beam.getFinalResult()
    
    def _beamSearchStep(self, beam, decoder_context, layout):
        pre_candidates = []
        # select each candidate
        for cur_cand_info in beam.candidates:
            # find beams
            decoder_input = Variable(torch.LongTensor([cur_cand_info.getCandidate()])).unsqueeze(0)
            decoder_input = decoder_input.cuda(self.gpu_id) if self.gpu_id != -1 else decoder_input
        
            decoder_output, decoder_hidden, attn_weight = self.decoder(decoder_input, cur_cand_info.hidden, decoder_context, layout)
            decoder_output = decoder_output.view(-1) # 1*1*10000 -> 10000
            
            # find candidates of candidates
            topVecs, topIdxs = decoder_output.sort(descending=True)
            pre_candidates += beam.electPreCandidates(topVecs, topIdxs, cur_cand_info, decoder_hidden, attn_weight.detach().squeeze(1).cpu())
            
        beam.electCandidates(pre_candidates)