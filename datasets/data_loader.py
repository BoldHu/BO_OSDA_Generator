import numpy as np
import torch
from torch.utils.data import Dataset
from utils.enumerator import SmilesEnumerator
from utils.utils import split, read_strings, read_vec
from utils.build_vocab import WordVocab
from rdkit import Chem
from tqdm import tqdm
import ast
import random
import pandas as pd

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

def smiles_to_seq(smile, vocab, seq_len=MAX_LEN):
    sm_spaced = split(smile) # Spacing
    sm_split = sm_spaced.split()
    if len(sm_split)<=MAX_LEN - 2:
        # convert to sequence by numpy
        content = [vocab.stoi.get(token, vocab.unk_index) for token in smile]
        X = [vocab.sos_index] + content + [vocab.eos_index]
        padding = [vocab.pad_index]*(seq_len - len(X))
        X.extend(padding)
        smiles_seq = np.array(X)
        return smiles_seq
    else:
        smile = split(smile).split()
        # convert to sequence by numpy
        content = [vocab.stoi.get(token, vocab.unk_index) for token in smile]
        X = [vocab.sos_index] + content + [vocab.eos_index]
        padding = [vocab.pad_index]*(seq_len - len(X))
        X.extend(padding)
        smiles_seq = np.array(X)
        return smiles_seq
    
class Contrastive_Seq2seqDataset(Dataset):
    def __init__(self, dataset, vocab, seq_len=220):
        self.zeo = dataset['zeo'].values
        self.syn = dataset['syn'].values
        self.smiles = dataset['smiles'].values
        self.positive_smiles = dataset['positive_smiles'].values
        self.canonical_smile_index = dataset['canonical_smile_index'].values
        # read numpy array as all_unique_smiles
        self.all_unique_smiles = np.load('./data/unique_smiles_seq.npy', allow_pickle=True)
        
        # convert zeo and syn to tensor
        self.zeo = [zeo.strip('[]').replace("'", '').split(', ') for zeo in tqdm(self.zeo)]
        self.zeo = np.array(self.zeo, dtype=np.float32)
        self.zeo = torch.tensor(self.zeo, dtype=torch.float32)
        
        self.syn = [syn.strip('[]').replace("'", '').split(', ') for syn in tqdm(self.syn)]
        self.syn = np.array(self.syn, dtype=np.float32)
        self.syn = torch.tensor(self.syn, dtype=torch.float32)

        # convert smiles to sequence and convert to tensor
        self.smiles = self.smiles.tolist()
        self.smiles = [smiles_to_seq(smile, vocab, seq_len) for smile in tqdm(self.smiles)]
        self.smiles = torch.tensor(self.smiles, dtype=torch.long)
        
        # convert positive smiles to sequence and convert to tensor
        # positive smiles is like <class 'numpy.ndarray'>: ["['CCO', 'CCC']", "['CCO', 'CCC']", "['CCO', 'CCC']"]
        self.positive_smiles = self.positive_smiles.tolist()
        self.positive_smiles = [ast.literal_eval(smile) for smile in self.positive_smiles]
        
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        zeo_item = self.zeo[item]
        syn_item = self.syn[item]
        sm_positive = self.positive_smiles[item]
        canonical_smile_index = self.canonical_smile_index[item]
        
        # convert positive smiles to sequence and convert to tensor
        sm_positive = [smiles_to_seq(smile, self.vocab, self.seq_len) for smile in sm_positive]
        sm_positive = torch.tensor(sm_positive, dtype=torch.long)
        
        # delete the canonical smile index from all_unique_smiles
        all_unique_smiles = np.delete(self.all_unique_smiles, canonical_smile_index, axis=0)
        # randomly select 64 negative samples
        sample_list = random.sample(range(len(all_unique_smiles)), 64)
        all_unique_smiles = all_unique_smiles[sample_list]
        # convert to tensor
        sm_negative = torch.tensor(all_unique_smiles, dtype=torch.long)
        
        return zeo_item, syn_item, sm, sm_positive, sm_negative
    
class Contrastive_Seq2seqDataset_random(Dataset):
    def __init__(self, dataset, vocab, seq_len=220):
        self.zeo = dataset['zeo'].values
        self.syn = dataset['syn'].values
        self.smiles = dataset['smiles'].values
        self.positive_smiles = dataset['positive_smiles'].values
        # read chembl smiles as all_unique_smiles and sample form it as negative samples
        self.all_unique_smiles = pd.read_csv('./data/chembl_24_remove.csv')['canonical_smiles'].values
        
        # convert zeo and syn to tensor
        self.zeo = [zeo.strip('[]').replace("'", '').split(', ') for zeo in tqdm(self.zeo)]
        self.zeo = np.array(self.zeo, dtype=np.float32)
        self.zeo = torch.tensor(self.zeo, dtype=torch.float32)
        
        self.syn = [syn.strip('[]').replace("'", '').split(', ') for syn in tqdm(self.syn)]
        self.syn = np.array(self.syn, dtype=np.float32)
        self.syn = torch.tensor(self.syn, dtype=torch.float32)

        # convert smiles to sequence and convert to tensor
        self.smiles = self.smiles.tolist()
        self.smiles = [smiles_to_seq(smile, vocab, seq_len) for smile in tqdm(self.smiles)]
        self.smiles = torch.tensor(self.smiles, dtype=torch.long)
        
        # convert positive smiles to sequence and convert to tensor
        # positive smiles is like <class 'numpy.ndarray'>: ["['CCO', 'CCC']", "['CCO', 'CCC']", "['CCO', 'CCC']"]
        self.positive_smiles = self.positive_smiles.tolist()
        self.positive_smiles = [ast.literal_eval(smile) for smile in self.positive_smiles]
        
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        zeo_item = self.zeo[item]
        syn_item = self.syn[item]
        sm_positive = self.positive_smiles[item]
        
        # convert positive smiles to sequence and convert to tensor
        sm_positive = [smiles_to_seq(smile, self.vocab, self.seq_len) for smile in sm_positive]
        sm_positive = torch.tensor(sm_positive, dtype=torch.long)
        
        # randomly select 64 negative samples
        negative_sample_index = random.sample(range(len(self.all_unique_smiles)), 64)
        negative_smiles = self.all_unique_smiles[negative_sample_index]
        # convert to sequence and convert to tensor
        sm_negative = [smiles_to_seq(smile, self.vocab, self.seq_len) for smile in negative_smiles]
        sm_negative = torch.tensor(sm_negative, dtype=torch.long)
        
        return zeo_item, syn_item, sm, sm_positive, sm_negative
    
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