import numpy as np
import pandas as pd
from rdkit import Chem

# extract the zeolite features in rows
def featurize_zeolite(row):
    features = [row['FD'], row['max_ring_size'], row['channel_dim'], row['inc_vol'], row['accvol'], row['maxarea'],
                row['minarea']]
    if any([pd.isnull(x) for x in features]):
        # print('Problem: ', row['Code'], 'Missing Information in Data')
        return None
    else:
        return np.array(features, dtype=float)

# extract the synthesis features in rows
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

# data augmentation by randomizing SMILES strings
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

# standardize the data
def norm(data):
    data_max, data_min = np.tile(np.max(data, axis=0), (data.shape[0], 1)), np.tile(
        np.min(data, axis=0), (data.shape[0], 1))
    data_norm = (data - data_min) / (data_max - data_min)
    return data_norm

# data filter
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
