import pandas as pd
from pandas import DataFrame as df
from rdkit import Chem
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from rdkit import Chem
import os
import sys

# set the path to the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from models.clamer import GptCovd
from configs.config_clamer import config
from utils.data_processing import data_augment, norm, data_sift
from datasets.data_loader import SampleDataset
from utils.utils import data_split

# data path
log_dir = './logs/'
save_best_weight_path = './checkpoints/'
sample_loss_history = []
now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
data_file = r'./data/OSDA_ZEO.xlsx'
model_file = r'./checkpoints/NO.0-2024-12-18-16-58-50-0.275966-0.15.pth'

# read data
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

# load model
model = GptCovd(d_model=config().d_model, 
                charlen=config().charlen,
                device=config().device,
                head=config().head,
                char_to_index=config().char_to_index).to(config().device)
model.load_state_dict(
    torch.load(model_file,  map_location=torch.device('cuda:0')))

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

        print('finished')
        bar.update()
        bar.close()

    writer.close()
    return sample_nll_total, smiles_gen_total


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
        'valid_smiles_null': valid_smiles_null, }).to_csv('./sample_result/AFI_gpt6lmpad_sample497.csv')  # aei_train_sample_gpt6lm02


