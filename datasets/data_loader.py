import numpy as np
import torch
from torch.utils.data import Dataset

# train dataset
class SeqDataset(Dataset):
    def __init__(self, zeo, syn, smis_seq):
        assert len(zeo) == len(syn) == len(smis_seq)
        # convert to tensor
        if not torch.is_tensor(zeo):
            # 先转化成浮点数,再转化成张量
            float_zeo = zeo.astype(np.float32)
            self.zeo = torch.tensor(float_zeo, dtype=torch.float32)
        else:
            self.zeo = zeo
            
        if not torch.is_tensor(syn):
            float_syn = syn.astype(np.float32)
            self.syn = torch.tensor(float_syn, dtype=torch.float32)
        else:
            self.syn = syn
        self.smis_seq = smis_seq

    def __getitem__(self, index):  # get sample pair
        return self.zeo[index], self.syn[index], self.smis_seq[index]

    def __len__(self):  # get sample_num
        return len(self.zeo)

# sample dataset
class SampleDataset(Dataset):
    def __init__(self, zeo, syn):
        assert len(zeo) == len(syn)  # == len(smis_seq)
        self.zeo = zeo
        self.syn = syn
        # self.smis_seq = smis_seq

    def __getitem__(self, index):  # get sample pair
        return self.zeo[index], self.syn[index]  # self.smis_seq[index]

    def __len__(self):  # get sample_num
        return len(self.zeo)