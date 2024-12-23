import torch

# set hyperparameters
class config():
    def __init__(self):
        self.batch_size = 64
        self.d_model = 128
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.charset = '?P25$]FO-S.Hc=71(ln63NC4[+)^@'
        self.charlen = len(self.charset)
        self.char_to_index = dict((c, i) for i, c in enumerate(self.charset))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.charset))
        self.head = 4
        self.seqlen = 127