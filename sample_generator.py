import pandas as pd
from pandas import DataFrame as df
from rdkit import Chem
import numpy as np
import random
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from rdkit import Chem


# 加载文件路径
log_dir = './logs/'
save_best_weight_path = './checkpoints/'
sample_loss_history = []
now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
data_file = r'./data/OSDA_ZEO.xlsx'
model_file = r'./checkpoints/NO.0-2024-12-18-16-58-50-0.275966-0.15.pth'

# 加载模型文件
from models.clamer import GptCovd
from configs.config_clamer import config

# 分子筛结构信息特征化
def featurize_zeolite(row):
    features = [row['FD'], row['max_ring_size'], row['channel_dim'], row['inc_vol'], row['accvol'], row['maxarea'],
                row['minarea']]
    if any([pd.isnull(x) for x in features]):
        # print('Problem: ', row['Code'], 'Missing Information in Data')
        return None
    else:
        return np.array(features, dtype=float)

# 合成信息特征化
def featurize_synthesis(row):
    seeds = ['seed', 'SAPO-56 seeds', 'SSZ-57', 'FAU', 'seeded with magadiite', 'seeds']
    solvents = ['ethylene glycol', 'hexanol', '2-propanol', 'triethylene glycol', 'triglycol',
                'polyethylene glycol', 'n-hexanol', 'glycol', 'propane-1,3-diol', 'butanol',
                'glycerol', 'isobutylamine', 'tetraethylene glycol', '1-hexanol',
                'sec-butanol', 'iso-butanol', 'ethylene glycol monomethyl ether', 'ethanol']
    acids = ['H2SO4', 'acetic acid', 'oxalic acid', 'succinic acid', 'arsenic acid', 'HNO3', 'HCl',
             'SO4']
    frameworks = ['Co', 'Mn', 'Cu', 'Zn', 'Cd', 'Cr', 'V', 'Ce', 'Nd', 'Sn', 'Zr', 'Ni',
                  'S', 'Sm', 'Dy', 'Y', 'La', 'Gd', 'In', 'Nb', 'Te', 'As', 'Hf', 'W',
                  'Se']
    common_frameworks = ['Si', 'Al', 'P', 'Ge', 'B', 'Ti', 'Ga', 'Fe']
    cations = ['Mg', 'Rb', 'Li', 'Cs', 'Sr', 'Ba', 'Be', 'Ca']
    common_cations = ['Na', 'K']
    bad = ['pictures', 'need access', 'also called azepane', 'SMILES code']
    syns = [x.strip() for x in [row['syn1'], row['syn2'], row['syn3'], row['syn4'], row['syn5'],
                                row['syn6'], row['syn7'], row['syn8']] if not pd.isnull(x)]
    if not syns:
        return None
    syn_vector = []
    for c in common_frameworks:
        if c in syns:
            syn_vector.append(1)
        else:
            syn_vector.append(0)
    for c in common_cations:
        if c in syns:
            syn_vector.append(1)
        else:
            syn_vector.append(0)
    if 'F' in syns:
        syn_vector.append(1)
    else:
        syn_vector.append(0)
    frame, cat, seed, solv, acid, oth = 0, 0, 0, 0, 0, 0
    for s in syns:
        if s in frameworks:
            frame = 1
        elif s in cations:
            cat = 1
        elif s in seeds:
            seed = 1
        elif s in solvents:
            solv = 1
        elif s in acids:
            acid = 1
        elif s.count(' ') < 2 and s not in bad and len(s) > 2:
            oth = 1
    syn_vector.extend([frame, cat, seed, solv, acid, oth])
    return np.array(syn_vector, dtype=float)

# 数据增强
def data_augment(augment, data):
    smiles_aug, zeo_features_aug, syn_features_aug, codes_aug = [], [], [], []
    for i, row in data.iterrows():
        if ' + ' not in row['smiles']:  # only look at single-template synthesis
            zeo = featurize_zeolite(row=row)
            syn = featurize_synthesis(row=row)
            if zeo is not None and syn is not None:
                if augment:
                    new_smiles = []
                    m = Chem.MolFromSmiles(row['smiles'])
                    for i in range(100):  # randomize smiles string up to 100 times
                        try:
                            rand_smile = Chem.MolToSmiles(m, canonical=False, doRandom=True, isomericSmiles=False)
                            rand_mol = Chem.MolFromSmiles(rand_smile)
                            if m is not None and rand_smile not in new_smiles:
                                new_smiles.append(rand_smile)
                        except:
                            # print('Problem:', row['smiles'], 'could not be randomized')
                            break
                    for smile in new_smiles:
                        smiles_aug.append(smile)
                        zeo_features_aug.append(zeo)
                        syn_features_aug.append(syn)
                        codes_aug.append(row['Code'])
                else:
                    smiles_aug.append(row['smiles'])
                    zeo_features_aug.append(zeo)
                    syn_features_aug.append(syn)
                    codes_aug.append(row['Code'])
    return smiles_aug, zeo_features_aug, syn_features_aug, codes_aug

# 归一化
def norm(data):
    data_max, data_min = np.tile(np.max(data, axis=0), (data.shape[0], 1)), np.tile(
        np.min(data, axis=0), (data.shape[0], 1))
    data_norm = (data - data_min) / (data_max - data_min)
    return data_norm

# 数据过滤
def data_sift(smiles, zeo_vecs, syn_vecs, codes):
    cant = ['a', 't', ' ', 'i', 'd', 'e', 'f', 'y', 'u']
    new_smile, new_zeo, new_syn, new_codes = [], [], [], []
    for s, z, v, d in zip(smiles, zeo_vecs, syn_vecs, codes):
        if 'Si' in s:
            continue
        found = False
        for c in s:
            if c in cant:
                found = True
        if not found:
            new_zeo.append(z)
            new_syn.append(v)
            new_smile.append(s)
            new_codes.append(d)

    return new_smile, np.array(new_zeo), np.array(new_syn), new_codes

# 数据字符串填充
def smiles_padding(smiles):
    src_smiles = []
    tgt_smiles = []
    start_char = '^'
    end_char = '$'
    pad_char = '?'
    for smis in smiles:
        pad_smile = start_char + smis + end_char + pad_char * 125
        src_smiles.append(pad_smile[0:125])
        tgt_smiles.append(pad_smile[0:126])
    return src_smiles, tgt_smiles

# 数据划分
def data_split(split, unique_codes, smiles, zeo_vectors, syn_vectors, codes):
    train_smiles, train_zeo, train_syn, train_codes = [], [], [], []
    test_smiles, test_zeo, test_syn, test_codes = [], [], [], []
    if split is None:
        train_smiles = smiles
        train_zeo = zeo_vectors
        train_syn = syn_vectors
        train_codes = codes
    elif split == 'random':
        unique_smiles = list(np.unique(smiles))
        print(len(unique_smiles))
        test_indices = []
        random.shuffle(unique_smiles)
        test_smiles_unique = unique_smiles[:round(0.2 * len(unique_smiles))]  # 20% held out set
        print(len(test_smiles_unique))
        for t in test_smiles_unique:
            for i, s in enumerate(smiles):
                if t == s:
                    test_indices.append(i)
        print(len(test_indices))
        for i, (s, z, v, c) in enumerate(zip(smiles, zeo_vectors, syn_vectors, codes)):
            if i in test_indices:
                test_smiles.append(s)
                test_zeo.append(z)
                test_syn.append(v)
                test_codes.append(c)
            else:
                train_smiles.append(s)
                train_zeo.append(z)
                train_syn.append(v)
                train_codes.append(c)
    else:
        if split not in unique_codes:
            print('Problem:', split, 'not a zeolite in the data')
        else:
            for i, (s, z, v, c) in enumerate(zip(smiles, zeo_vectors, syn_vectors, codes)):
                if split == c:
                    test_smiles.append(s)
                    test_zeo.append(z)
                    test_syn.append(v)
                    test_codes.append(c)
                else:
                    train_smiles.append(s)
                    train_zeo.append(z)
                    train_syn.append(v)
                    train_codes.append(c)
    return train_smiles, train_zeo, train_syn, train_codes, test_smiles, test_zeo, test_syn, test_codes

# 数据集预制
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

# 设置采样机制
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

# 生成函数
def sample_gpt(epochs, temp):
    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
    sample_nll_total = []
    smiles_gen_total = []
    for epoch in range(epochs):
        bar = tqdm(total=len(sample_dataloader), ncols=125)
        bar.set_description_str(f"{epoch}/{epochs}")

        for batch_idx, (zeo, syn) in enumerate(sample_dataloader):
            with torch.no_grad():
                target = [config().char_to_index['^']] + [config().char_to_index['?']] * 124
                tgt_seq = torch.LongTensor(target).unsqueeze(0).expand(zeo.size(0), len(target)).to(config().device)

                zeo, syn = zeo.to(config().device), syn.to(config().device)
                smiles_gen = [[''] * config().batch_size][0]
                sample_nll = [0] * config().batch_size
                finished = np.array([False] * config().batch_size, dtype=object)
                end_char = '$'
                for i in range(124):
                    net_out = model(zeo, syn, tgt_seq)[:, i + 2, :]
                    o = F.softmax(net_out, dim=-1).cpu().detach().numpy()
                    # sample temp
                    if temp != 0:
                        temp = abs(temp)  # No negative values
                        next_char_probs = np.log(o) / temp
                        next_char_probs = np.exp(next_char_probs)
                        next_char_probs = next_char_probs.astype(float)
                        next_char_probs = (next_char_probs.T / (next_char_probs.sum(axis=1))).T
                        sampleidc = torch.tensor(
                            [np.random.multinomial(1, next_char_prob, 1).argmax() for next_char_prob in
                             next_char_probs])
                    else:
                        sampleidc = torch.tensor(np.argmax(o, axis=1))

                    samplechars = [config().index_to_char[idx] for idx in sampleidc.numpy()]

                    for idx, samplechar in enumerate(samplechars):
                        if not finished[idx]:
                            if samplechar != end_char:
                                # Append the SMILES with the next character
                                smiles_gen[idx] += samplechar
                                tgt_seq[:, i + 1] = sampleidc.to(config().device)
                                # Calculate negative log likelihood for the selected character
                                sample_nll[idx] -= np.log(o[idx][sampleidc[idx]])
                            else:
                                finished[idx] = True
                                # print("SMILES has finished at %i" %i)
                    # If all SMILES are finished, i.e. the end_char "$" has been generated, stop the generation
                if finished.sum() == len(finished):
                    sample_nll_total += sample_nll
                    smiles_gen_total += smiles_gen
                bar.update()

        bar.update()
        bar.close()

    writer.close()
    return sample_nll_total, smiles_gen_total

if __name__ == '__main__':
    # 读取数据
    data = pd.read_excel(data_file, engine='openpyxl')
    smiles_aug, zeo_features_aug, syn_features_aug, codes_aug = data_augment(augment=False, data=data)

    zeo_features = np.array(zeo_features_aug)
    zeo_features = norm(zeo_features)

    smiles, zeo_vectors, syn_vectors, codes = data_sift(smiles_aug, zeo_features, syn_features_aug, codes_aug)
    print(len(set(smiles)), len(set(codes)))
    unique_codes = ['AFI']
    split = "AFI"
    _, _, _, _, test_smiles, test_zeo, test_syn, _ = data_split(split, unique_codes,
                                                                         smiles, zeo_vectors, syn_vectors, codes)
    test_smiles = test_smiles * 250
    test_zeo = test_zeo * 250  # * 121 42
    test_syn = test_syn * 250  # * 121 42
    ref_smiles = test_smiles[0:1000]
    sample_zeo = test_zeo[0:1000]
    sample_syn = test_syn[0:1000]
    sample_dataset = SampleDataset(sample_zeo, sample_syn)
    # print('samples of testset without augmented is: ', len(train_dataset))
    sample_dataloader = DataLoader(sample_dataset, batch_size=config().batch_size, shuffle=False, collate_fn=None, pin_memory=True)
    # 加载模型
    model = GptCovd(d_model=config().d_model, 
                    charlen=config().charlen,
                    device=config().device,
                    head=config().head,
                    char_to_index=config().char_to_index).to(config().device)
    model.load_state_dict(
        torch.load(model_file,  map_location=torch.device('cuda:0')))

    # 生成采样
    sam_null, sam_text = sample_gpt(1, 1)
    # 评估分子有效性
    valid_smiles_ref = []
    valid_smiles_sam = []
    valid_smiles_null = []
    valid_zeos = []
    valid_syns = []
    for i, (r, s, n, z, y) in enumerate(zip(ref_smiles, sam_text, sam_null, sample_zeo, sample_syn)):
        try:
            mol_ge = Chem.MolFromSmiles(s)
            if mol_ge is not None:
                valid_smiles_sam.append(s)
                valid_smiles_null.append(n)
                valid_smiles_ref.append(r)
                valid_zeos.append(z)
                valid_syns.append(y)

        except:
            pass
    print('total valid smiles num: ', len(valid_smiles_sam))
    print('unique smilee num: ', len(set(valid_smiles_sam)))
    df({'mol_gen': valid_smiles_sam, 'smiles': valid_smiles_ref, 'zeos': valid_zeos, 'syns': valid_syns,
            'valid_smiles_null': valid_smiles_null, }).to_csv('AFI_gpt6lmpad_sample497.csv')  # aei_train_sample_gpt6lm02
