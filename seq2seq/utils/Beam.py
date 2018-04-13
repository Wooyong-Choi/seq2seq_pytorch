class Beam(object):
    """
    For beam search
    """
    def __init__(self, beam_size, init_hidden, SOS_IDX, EOS_IDX):
        super(Beam, self).__init__()
        self.step = 1
        self.beam_size = beam_size
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX
        
        self.candidates = [Candidate([self.sos_idx], 0, init_hidden)]
        self.eos_best = Candidate(None, float('-inf'), None)
        self.early_end = None
        
    def electPreCandidates(self, topVecs, topIdxs, cur_cand_info, hidden):
        pre_candidates = []
            
        for next, next_score in zip(topIdxs.data, topVecs.data):
            # Append beams to score board
            pre_cand = Candidate(cur_cand_info.seq + [next], cur_cand_info.score + next_score, hidden)
            
            if len(pre_candidates) >= self.beam_size:
                break
            elif pre_cand.endWithEOS(self.eos_idx):
                self.eos_best = pre_cand if self.eos_best.score < pre_cand.score else self.eos_best
                
            pre_candidates.append(pre_cand)  # elected
            
        return pre_candidates
                
    def electCandidates(self, pre_candidates):
        pre_candidates = sorted(pre_candidates, key=lambda x: x.score, reverse=True)
        
        # if first ranked sequence is ended with EOS_IDX, return it
        if pre_candidates[0].endWithEOS(self.eos_idx):
            self.early_end = pre_candidates[0]
        
        # reset every step
        self.candidates = []
        for pre_cand in pre_candidates:
            if pre_cand.endWithEOS(self.eos_idx):
                continue
            
            self.candidates.append(pre_cand)
            
            if len(self.candidates) >= self.beam_size:
                break
            
        self.step += 1
        
    def getFirstRanked(self):
        return self.candidates[0]
        
    def isEarlyEnd(self):
        return self.eos_best.hidden is None
    
    def getFinalResult(self):
        if self.eos_best.seq is not None:
            return self.eos_best
        else:
            return self.getFirstRanked()  # error: no seq endded with eos
        
    def __repr__(self):
        str = "Step : {}\n".format(self.step)
        str += "candidates\n"
        for cand in self.candidates:
            str += cand.__repr__() + '\n'
        str += 'eos_best\n'+self.eos_best.__repr__() + '\n'
        return str

class Candidate(object):
    """
    A wrapper class for beam searching
    """
    def __init__(self, seq, score, hidden):
        super(Candidate, self).__init__()
        
        self.seq = seq
        self.score = score
        self.hidden = hidden
    
    def getCandidate(self):
        return self.seq[-1]
    
    def endWithEOS(self, eos_idx):
        return self.seq[-1] == eos_idx
        
    def __repr__(self):
        return "Seq : {} \nScore : {}\n".format(self.seq, self.score)