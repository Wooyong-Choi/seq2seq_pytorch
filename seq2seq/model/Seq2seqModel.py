import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
from seq2seq.utils import Beam

class Seq2seqModel(nn.Module):
    def __init__(self, name, input_size, hidden_size, output_size, max_length, gpu_id=-1):
        super(Seq2seqModel, self).__init__()
        self.name = name
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.max_length = max_length
        self.gpu_id = gpu_id
        
        """
        Encoder uses GRU with embedding layer and packing sequence
        """
        self.encoder = EncoderRNN(input_size, hidden_size)
        
        """
        Decoder uses GRU with embedding layer, fully connected layer and log-softmax
        """
        self.decoder = DecoderRNN(hidden_size, output_size)

        if self.gpu_id != -1:
            self.cuda(self.gpu_id)
            
    def forward(self, encoder_inputs, decoder_inputs, encoder_inputs_lengths):
        """
        Params:
        -------
        encoder_input: sequence of word indices,  "<SOS> How are you? <PAD>"
        decoder_input: sequence of word indices,  "<SOS> I'm fine. <PAD>"
        decoder_target: sequence of word indices, "I'm fine. <EOS> <PAD>"
        encoder_inputs_lengths: lengths of encoder inputs to use pack_padded_sequence
        
        Returns:
        --------
        decoder_outputs : teacher forced outputs of decoder
        #loss: NLL loss of log-softmax value of generated and target sequence.
        """
            
        cur_batch_size = encoder_inputs.size()[0]  # batch_first
        
        # Encoder : sentence -> context
        encoder_hidden = self.encoder.initHidden(cur_batch_size, self.gpu_id)
        encoder_outputs, encoder_hidden = self.encoder(encoder_inputs, encoder_inputs_lengths, encoder_hidden)
        
        # Decoder : context -> response
        decoder_hidden = encoder_hidden
        decoder_outputs, decoder_hidden = self.decoder(decoder_inputs, decoder_hidden)
        
        return decoder_outputs
    
    def sampleResponce(self, sentence, tagger, src_vocab, tgt_vocab, beam_size=5):
        self.eval()
        
        encoder_input = torch.LongTensor(src_vocab.sequence_to_indices(sentence)).unsqueeze(0)
        encoder_input = Variable(encoder_input)
        encoder_input = encoder_input.cuda(self.gpu_id) if self.gpu_id != -1 else encoder_input
        encoder_input_length = [encoder_input.size()[1]]
        
        # Encoder
        encoder_hidden = self.encoder.initHidden(1, gpu_id=self.gpu_id)  # batch_size = 1
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_input_length, encoder_hidden)
    
        # Decoder
        decoder_hidden = encoder_hidden
        beam_result = self._decodeWithBeamSearch(decoder_hidden, beam_size, tgt_vocab.sos_idx, tgt_vocab.eos_idx)

        # indices -> word sequence
        decoded_words = []
        for idx in beam_result.seq:
            if idx == tgt_vocab.eos_idx:
                decoded_words.append(tgt_vocab.eos_tok)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[idx])
                
        self.train()
        return decoded_words
    
    def _decodeWithBeamSearch(self, decoder_hidden, beam_size, SOS_IDX, EOS_IDX):
        """
        Find best sequence using beam search
        """
        # list of [[sequence] and score]
        beam_board = [Beam([SOS_IDX], 0)]
        eos_best = Beam(None, float('inf'))   # neg inf score
                
        # with beam search
        for i in range(self.max_length+1):
            # beam result can be searched beam board or final result of beam search
            decoder_hidden, beam_result, eos_result = self._beamSearchStep(decoder_hidden, beam_board, beam_size, EOS_IDX);
            
            # if final result, return it
            if isinstance(beam_result, Beam):
                return beam_result
            
            if eos_result is not None:
                eos_best = eos_result if eos_best.score < eos_result.score else eos_best
                
            # select top 5 candidates in our_socre_board
            beam_board = sorted(beam_result, key=lambda x: x.score, reverse=True)[:beam_size]
        
        # find beam which has EOS_IDX at the end
        for beam in beam_board:
            if beam.seq[-1] == EOS_IDX:
                return beam.seq
            
        # if there's no such beam, return eos_best
        if eos_best.seq is not None:
            return eos_best.seq
        # failure case, no sequence ended with eos
        else:
            return beam_board[0].seq
    
    def _beamSearchStep(self, decoder_hidden, beam_board, beam_size, EOS_IDX):
        cur_beam_board = []
        
        # keep sequence which end with eos
        cur_eos_best = Beam([], -sys.maxsize-1)   # neg inf score
        
        # select each candidate
        for cur_beam in beam_board: # [[sequence], score]
            cur_seq = cur_beam.seq
            cur_score = cur_beam.score
            
            candidate = cur_seq[-1]
            
            # find beams
            decoder_input = Variable(torch.LongTensor([[candidate]]))
            decoder_input = decoder_input.cuda(self.gpu_id) if self.gpu_id != -1 else decoder_input
        
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = decoder_output.view(-1) # 1*1*10000 -> 10000
            
            # find candidates of candidates
            topVecs, topIdxs = decoder_output.topk(beam_size)
            
            # if first ranked next word is eos, stop beam search
            if topIdxs.data[0] == EOS_IDX:
                return decoder_hidden, Beam(cur_seq+[EOS_IDX], 0), None
              
            for next, next_score in zip(topIdxs.data, topVecs.data):    
                # Append beams to score board
                seq = cur_seq + [next]
                score = cur_score + next_score  # log softmax
                beam = Beam(seq, score)
                
                if next == EOS_IDX:
                    cur_eos_best = beam if cur_eos_best.score < beam.score else cur_eos_best
                
                cur_beam_board.append(beam)
                
        return decoder_hidden, cur_beam_board, cur_eos_best