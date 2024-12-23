# calculate the metrics for the molecular generation task
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import linalg

# calculate the Validity
def validity_rate(smiles):
    valid = 0
    # show the percentage of checking process
    for i, smile in enumerate(smiles):
        if i % 100 == 0:
            print('Checking the validity of %dth smile' % i / len(smiles))
        if Chem.MolFromSmiles(smile):
            valid += 1
    return valid / len(smiles)

# calculate the Uniqueness
def uniqueness_rate(smiles):
    return len(set(smiles)) / len(smiles)

# calculate the Reconstructability: the proportion of unique molecules generated having appered in the dataset
def reconstructability_rate(smiles, data):
    data_smiles = data['smiles'].tolist()
    unique_smiles = set(smiles)
    reconstructable = 0
    for smile in unique_smiles:
        if smile in data_smiles:
            reconstructable += 1
    return reconstructable / len(unique_smiles)

# calculate the FCD score: obtained by calculating the distribution of features between the real and generated molecules
def FCD_score(real_data, generated_data):
    real_data = np.array(real_data)
    generated_data = np.array(generated_data)
    mu1 = np.mean(real_data, axis=0)
    mu2 = np.mean(generated_data, axis=0)
    sigma1 = np.cov(real_data, rowvar=False)
    sigma2 = np.cov(generated_data, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    eps = 1e-6
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# calculate the IntDiv (Internal diversity): measuring whether the model continuously generates structurally similar molecules
def IntDiv(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    fps = np.array([list(fp.ToBitString()) for fp in fps], dtype=int)
    return np.mean(np.std(fps, axis=0))

