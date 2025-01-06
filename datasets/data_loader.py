import numpy as np
import torch
from torch.utils.data import Dataset
from utils.enumerator import SmilesEnumerator
from utils.utils import *
from utils.build_vocab import WordVocab
from rdkit import Chem
PAD = 0
MAX_LEN = 220

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


class Randomizer(object):

    def __init__(self):
        self.sme = SmilesEnumerator()
    
    def __call__(self, sm):
        sm_r = self.sme.randomize_smiles(sm) # Random transoform
        if sm_r is None:
            sm_spaced = split(sm) # Spacing
        else:
            sm_spaced = split(sm_r) # Spacing
        sm_split = sm_spaced.split()
        if len(sm_split)<=MAX_LEN - 2:
            return sm_split # List
        else:
            return split(sm).split()

    def random_transform(self, sm):
        '''
        function: Random transformation for SMILES. It may take some time.
        input: A SMILES
        output: A randomized SMILES
        '''
        return self.sme.randomize_smiles(sm)

class Seq2seqDataset(Dataset):

    def __init__(self, zeo, syn, smiles, vocab, seq_len=220, transform=Randomizer()):
        assert len(zeo) == len(syn) == len(smiles)
        # convert to tensor
        if not torch.is_tensor(zeo):
            # convert to float first, then convert to tensor
            float_zeo = zeo.astype(np.float32)
            self.zeo = torch.tensor(float_zeo, dtype=torch.float32)
        else:
            self.zeo = zeo
            
        if not torch.is_tensor(syn):
            float_syn = syn.astype(np.float32)
            self.syn = torch.tensor(float_syn, dtype=torch.float32)
        else:
            self.syn = syn
        # convert SMILES to sequence from np.array to list
        if type(smiles) == np.ndarray:
            smiles = smiles.tolist()
            # convert it from [[''], [''], ['']] to ['','','']
            smiles = [i[0] for i in smiles]
        self.smiles = smiles
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        sm = self.transform(sm) # List
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index]*(self.seq_len - len(X))
        X.extend(padding)
        smiles_seq = torch.tensor(X)
        return self.zeo[item], self.syn[item], smiles_seq

if __name__ == '__main__':
    # read the data and convert to the format we need
    train_smiles = read_strings('./data/train_smiles.csv', idx=False)
    train_zeo = read_vec('./data/train_zeo.csv', idx=False)
    train_syn = read_vec('./data/train_syn.csv', idx=False)
    train_codes = read_strings('./data/train_codes.csv', idx=False)
    test_smiles = read_strings('./data/test_smiles.csv', idx=False)
    test_zeo = read_vec('./data/test_zeo.csv', idx=False)
    test_syn = read_vec('./data/test_syn.csv', idx=False)
    test_codes = read_strings('./data/test_codes.csv', idx=False)

    vocab = WordVocab.load_vocab('./model_hub/vocab.pkl')
    print('the vocab size is :', len(vocab))

    charlen = len(vocab)
    print('the total num of charset is :', charlen)
    
    # create the dataset and dataloader
    train_dataset = Seq2seqDataset(train_zeo, train_syn, train_smiles, vocab)
    test_dataset = Seq2seqDataset(test_zeo, test_syn, test_smiles, vocab)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    print('the train dataset length is :', len(train_dataset))
    print('the test dataset length is :', len(test_dataset))
    for i, (zeo, syn, smis) in enumerate(train_loader):
        print('the zeo shape is :', zeo.shape)
        print('the syn shape is :', syn.shape)
        print('the smis shape is :', smis.shape)
        print('the smis is :', smis)
        break