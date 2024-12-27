import csv
import numpy as np
import torch
import random
import torch
import torch.nn.functional as F

def read_vec(filepath, idx=True):  # 定义文件名称，本文件要与当前的.py文件要在同一文件夹下，不然要用绝对路径
    if idx:
        with open(filepath, 'r') as csvfile:  # 打开数据文件
            reader = csv.reader(csvfile)  # 用csv的reader函数读取数据文件
            header = next(reader)  # 读取数据文件的表头
            data = []  # 定义一个空数组用于保存文件的数据
            for line in reader:  # 循环读取数据文件并保存到数组data中
                for i, s in enumerate(line):
                    line[i] = float(s)
                data.append(line)  # line是个一维数组，是数据文件中的一行数据
        return np.array(data)[:, 1:]
    else:
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            data = []
            for line in reader:
                data.append(line)
        return np.array(data)

# 读取生成的smiles字符串csv文件（源数据的whim-pca数据）
def read_strings(filepath, idx=True):
    if idx:
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            data = []
            for line in reader:
                data.append(line)
        return np.array(data)[:, 1:]
    else:
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            data = []
            for line in reader:
                data.append(line)
        return np.array(data)

# 字符串填充
def smiles_padding(smiles):
    src_smiles = []
    tgt_smiles = []
    start_char = '^'
    end_char = '$'
    pad_char = '?'
    for smis in smiles:
        if isinstance(smis, np.ndarray):
            smis = smis[0]
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
        print('Number of unique smiles:', len(unique_smiles))
        print(len(unique_smiles))
        test_indices = []
        random.shuffle(unique_smiles)
        test_smiles_unique = unique_smiles[:round(0.2 * len(unique_smiles))]  # 20% held out set
        print(len(test_smiles_unique))
        # for t in test_smiles_unique:
        #     for i, s in enumerate(smiles):
        #         if t == s:
        #             test_indices.append(i)
        for idx, t in enumerate(test_smiles_unique):
            for i, s in enumerate(smiles):
                if t == s:
                    test_indices.append(i)
            print('进度:', idx, '/', len(test_smiles_unique))
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

# convert the SMILES to sequence
def smiles_to_sequence(smis_list, char_to_index):
    smis_list_sequence = []
    for smis in smis_list:
        smis_seq = []
        for c in smis:
            smis_seq.append(char_to_index.get(c))
        smis_seq = torch.tensor(smis_seq, dtype=torch.float32)
        smis_list_sequence.append(smis_seq)
    return smis_list_sequence

# convert the sequence to SMILES
def sequence_to_smiles(seq_list, index_to_char):
    # check if the sequence is a tensor
    if isinstance(seq_list, torch.Tensor):
        seq_list = seq_list.tolist()
    elif isinstance(seq_list, np.ndarray):
        seq_list = seq_list.tolist()
    smis_list = []
    for seq in seq_list:
        smis = ''
        for i in seq:
            element = index_to_char[int(i)]
            if element == '$':
                break
            elif element == '^':
                continue
            else:
                smis += element
        smis_list.append(smis)
    return smis_list

# set the top k logits to -inf
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

# predict the SMILES
def predict(zeo, syn, model, char_to_index):
    # x = [1, 50]
    model.eval()

    target = [char_to_index['^']] + [char_to_index['?']] * 124
    target = torch.LongTensor(target).unsqueeze(0)
    # search for the next character in the sequence
    for i in range(124):
        # [1, 50]
        out = F.softmax(model(zeo, syn, target)[:, i + 2, :])
        out = out.argmax(dim=1).detach()
        # add the predicted character to the sequence
        target[:, i + 1] = out

    return target