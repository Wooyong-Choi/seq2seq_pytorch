import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

from seq2seq.model import EncoderRNN
from seq2seq.model import DecoderRNN
from seq2seq.utils import Beam
from seq2seq.data import SOS_IDX, EOS_IDX

class Seq2seqModel(nn.Module):
    """
    A seq2seq model has encoder and encoder using RNN.
    """
    
    def __init__(self, n_layers, input_size, emb_size, hidden_size, output_size, max_tgt_len, dropout_p, bi_encoder, device):
        
        super(Seq2seqModel, self).__init__()
        
        self.n_layers = n_layers
        self.input_size = input_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.bi_encoder = bi_encoder
        self.dropout_p = dropout_p
        
        self.max_tgt_len = max_tgt_len
        
        self.device = device
        
        """
        Encoder uses GRU with embedding layer and packing sequence
        """
        self.encoder = EncoderRNN(n_layers,
                                  input_size,
                                  emb_size,
                                  hidden_size,
                                  dropout_p,
                                  bi_encoder)
        
        """
        Decoder uses GRU with embedding layer, fully connected layer and log-softmax
        """
        self.decoder = DecoderRNN(n_layers,
                                  hidden_size*2 if bi_encoder else hidden_size,
                                  emb_size,
                                  output_size,
                                  dropout_p)
        
        self.to(device)
            
    def forward(self, src_batch, tgt_batch, src_batch_lengths):
        """
        Params:
        -------
        src_batch: sequence of word indices,  "How are you? <EOS> <PAD>"
        tgt_batch: sequence of word indices,  "<SOS> I'm fine. <PAD>"
        src_batch_lengths: lengths of source batch to use pack_padded_sequence
        tgt_batch_lengths: lengths of target batch to use masked cross entropy
        
        Returns:
        --------
        decoder_outputs : teacher forced outputs of decoder
        """
        
        # Encoder : sentence -> context
        encoder_hidden = self.initHidden(src_batch)
        encoder_outputs, encoder_hidden = self.encoder(src_batch, src_batch_lengths, encoder_hidden)
        
        # Decoder : context -> response
        decoder_hidden = self._cat_directions(encoder_hidden) if self.bi_encoder else encoder_hidden
        decoder_context = encoder_outputs
        decoder_outputs, decoder_hidden, attn_weights = self.decoder(tgt_batch, decoder_hidden, decoder_context)
        
        return decoder_outputs
    
    def initHidden(self, src_batch):
        num_direction = 2 if self.bi_encoder else 1
        cur_batch_size = src_batch.size(0) # batch_first
        
        result = torch.zeros(self.n_layers * num_direction, cur_batch_size, self.hidden_size).to(self.device)
        return result
    
    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
    
    def sampleResponce(self, indices, src_vocab, tgt_vocab, beam_size=-1):
        self.eval()
        
        encoder_input = torch.tensor(indices).unsqueeze(0)
        encoder_input = encoder_input.to(self.device)
        encoder_input_length = [encoder_input.size()[1]]
        
        # Encoder
        encoder_hidden = self.initHidden(torch.zeros(1))
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_input_length, encoder_hidden)
    
        # Decoder
        decoder_hidden = self._cat_directions(encoder_hidden) if self.bi_encoder else encoder_hidden
        decoder_context = encoder_outputs
        
        decoded = []
        
        decoded_words = []
        attn_weights = []
        
        # top 1 decoder
        if beam_size == -1 or beam_size == 1:
            decoded = [SOS_IDX]
            for i in range(self.max_tgt_len+1):
                decoder_input = torch.tensor([decoded[-1]]).unsqueeze(0)
                decoder_input = decoder_input.to(self.device)
                
                decoder_output, decoder_hidden, attn_weight = self.decoder(decoder_input, decoder_hidden, decoder_context)
                attn_weights.append(attn_weight.detach().squeeze(1).cpu())
                decoder_output = decoder_output.view(-1) # 1*1*10000 -> 10000
                
                # find candidates of candidates
                topVecs, topIdxs = decoder_output.topk(1)
                decoded.append(topIdxs.item())
                
                if topIdxs.item() == EOS_IDX:
                    break
            
            attn_weights = torch.cat(attn_weights, dim=0)
            
        # top k decoder with beam search
        else:
            elected_cand = self._decodeWithBeamSearch(decoder_hidden, decoder_context, beam_size)
            decoded = elected_cand.seq
            attn_weights = elected_cand.attn
            
        # indices -> word sequence
        for idx in decoded[1:]:
            decoded_words.append(tgt_vocab.index2word[idx])
        
        self.train()
        return decoded_words, attn_weights
    
    def _decodeWithBeamSearch(self, decoder_hidden, decoder_context, beam_size):
        """
        Find best sequence using beam search
        """
        beam = Beam(beam_size, decoder_hidden, SOS_IDX, EOS_IDX)

        # with beam search
        for i in range(self.max_tgt_len+1):
            # beam result can be searched beam board or final result of beam search
            self._beamSearchStep(beam, decoder_context)
            # if first ranked sequence is ended with EOS_IDX, return it
            if beam.early_end is not None:
                return beam.early_end
        # no early end
        return beam.getFinalResult()
    
    def _beamSearchStep(self, beam, decoder_context):
        pre_candidates = []
        # select each candidate
        for cur_cand_info in beam.candidates:
            # find beams
            decoder_input = torch.tensor([cur_cand_info.getCandidate()]).unsqueeze(0)
            decoder_input = decoder_input.to(self.device)
        
            decoder_output, decoder_hidden, attn_weight = self.decoder(decoder_input, cur_cand_info.hidden, decoder_context)
            decoder_output = decoder_output.view(-1) # 1*1*10000 -> 10000
            
            # find candidates of candidates
            topVecs, topIdxs = decoder_output.sort(descending=True)
            pre_candidates += beam.electPreCandidates(topVecs, topIdxs, cur_cand_info, decoder_hidden, attn_weight.detach().squeeze(1).cpu())
            
        beam.electCandidates(pre_candidates)
