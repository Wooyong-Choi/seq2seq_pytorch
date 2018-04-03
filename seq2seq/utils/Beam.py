class Beam(object):
    def __init__(self, seq, score):
        super(Beam, self).__init__()
        
        self.seq = seq
        self.score = score
        
    def __repr__(self):
        return "Seq : {} \nScore : {}\n".format(self.seq, self.score)