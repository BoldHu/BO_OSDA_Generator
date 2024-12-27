# calculate the metrics for the molecular generation task
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import fcd
from scipy.stats import entropy

# calculate the Validity
def validity_rate(smiles):
    valid = 0
    # show the percentage of checking process
    for i, smile in enumerate(smiles):
        if Chem.MolFromSmiles(smile):
            valid += 1
    return valid / len(smiles)

# calculate the Uniqueness
def uniqueness_rate(smiles):
    return len(set(smiles)) / len(smiles)

# calculate the Reconstructability: the proportion of unique molecules generated having appered in the dataset
def reconstructability_rate(smiles, data_smiles):
    unique_smiles = set(smiles)
    reconstructable = 0
    for smile in unique_smiles:
        if smile in data_smiles:
            reconstructable += 1
    return reconstructable / len(unique_smiles)

# calculate the Novelty: the proportion of unique molecules generated not in the dataset
def novelty_rate(smiles, data_smiles):
    unique_smiles = set(smiles)
    novel = 0
    for smile in unique_smiles:
        if smile not in data_smiles:
            novel += 1
    return novel / len(unique_smiles)

# calculate the FCD score: obtained by calculating the distribution of features between the real and generated molecules
def FCD_score(real_data, generated_data):
    fcd_chemnet = fcd.load_ref_model()
    
    # filter the invalid smiles and get canonical smiles
    real_data_valid = []
    for smile in real_data:
        if Chem.MolFromSmiles(smile):
            mol = Chem.MolFromSmiles(smile)
            real_data_valid.append(Chem.MolToSmiles(mol, canonical=True))
    generated_data_valid = []
    for smile in generated_data:
        if Chem.MolFromSmiles(smile):
            mol = Chem.MolFromSmiles(smile)
            generated_data_valid.append(Chem.MolToSmiles(mol, canonical=True))
    
    # calculate the FCD score
    fcd_score = fcd.get_fcd(real_data, generated_data, fcd_chemnet)
    return fcd_score

# calculate the IntDiv (Internal diversity): measuring whether the model continuously generates structurally similar molecules
def IntDiv(smiles_list):
    # convert SMILES to RDKit molecules
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list if Chem.MolFromSmiles(smiles)]
    
    # generate Morgan fingerprints for each molecule
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in mols]
    
    # calculate the Tanimoto similarity between each pair of molecules
    num_mols = len(fps)
    tanimoto_similarities = []
    
    for i in range(num_mols):
        for j in range(i + 1, num_mols):  # only calculate the upper triangle
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            tanimoto_similarities.append(sim)
    
    # calculate the IntDiv: the average Tanimoto similarity between all pairs of molecules
    if tanimoto_similarities:
        avg_similarity = np.mean(tanimoto_similarities)
        intdiv = 1 - avg_similarity
    else:
        intdiv = 0.0  # if no valid molecules, return 0
    
    return intdiv

def KL_divergence(real_smiles, generated_smiles, radius=2, n_bits=2048, bins=20):
    """
    Compute the KL divergence between the distributions of molecular fingerprints
    of two sets of SMILES strings (real and generated molecules).

    Parameters:
    - real_smiles: list of SMILES strings for the real molecules.
    - generated_smiles: list of SMILES strings for the generated molecules.
    - radius: Radius of the Morgan fingerprint (default: 2).
    - n_bits: Number of bits for the fingerprint (default: 2048).
    - bins: Number of bins to discretize the fingerprint similarity scores (default: 20).

    Returns:
    - kl_div: The KL divergence value.
    """

    def compute_fingerprint_distribution(smiles_list):
        """
        Compute the distribution of fingerprint similarity scores for a given set of SMILES strings.

        Parameters:
        - smiles_list: list of SMILES strings.

        Returns:
        - fingerprint_distribution: Histogram distribution of fingerprint similarities.
        """
        # Generate Morgan fingerprints for the molecules
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius, nBits=n_bits)
                        for smiles in smiles_list if Chem.MolFromSmiles(smiles)]

        # Calculate pairwise similarity scores (Tanimoto similarity)
        num_fps = len(fingerprints)
        similarity_scores = []
        for i in range(num_fps):
            for j in range(i + 1, num_fps):  # Avoid duplicate pairs
                sim = AllChem.DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                similarity_scores.append(sim)

        # Compute the histogram (probability distribution) of similarity scores
        histogram, bin_edges = np.histogram(similarity_scores, bins=bins, range=(0, 1), density=True)

        # Smooth the histogram by adding a small epsilon to avoid zero probabilities
        epsilon = 1e-10
        histogram += epsilon

        # Normalize the histogram to ensure it sums to 1
        histogram /= np.sum(histogram)

        return histogram

    # Compute fingerprint distributions for real and generated molecules
    real_distribution = compute_fingerprint_distribution(real_smiles)
    generated_distribution = compute_fingerprint_distribution(generated_smiles)

    # Compute KL divergence between the two distributions
    kl_div = entropy(real_distribution, generated_distribution)

    return kl_div

if __name__ == '__main__':
    original_data = ['n1(C)c(C)[n+](cc1)Cc1ccccc1C[n+]1ccn(C)c1C', 
                     'Cn1cc[n+](Cc2ccccc2C[n+]2ccn(C)c2C)c1C',
                     'Cc1n(C)cc[n+]1Cc1c(cccc1)C[n+]1ccn(c1C)C',
                     'c1ccc(c(c1)C[n+]1c(C)n(C)cc1)C[n+]1ccn(c1C)C',
                     '[n+]1(c(n(C)cc1)C)Cc1c(C[n+]2ccn(c2C)C)cccc1',
                     'Cc1n(C)cc[n+]1Cc1ccccc1C[n+]1c(n(cc1)C)C',
                     'Cn1c(C)[n+](cc1)Cc1ccccc1C[n+]1ccn(c1C)C',
                     'c1c[n+](Cc2ccccc2C[n+]2c(C)n(cc2)C)c(C)n1C',
                     'n1(cc[n+](c1C)Cc1ccccc1C[n+]1ccn(C)c1C)C',
                     '[n+]1(Cc2c(C[n+]3c(n(cc3)C)C)cccc2)c(C)n(C)cc1']
    
    generated_data = ['C(CCCCCCCCCC[N+](C)(C)C)(C)C', 
                      'c1cn(C[n+]2c(ccccc2)C)C1c(n1)C', 
                      '[n+]1(ccn(c1C)C)CCCC[n+]1cc(n(C)cc1C)C', 
                      'c1ccc(C[n+]2cc(c2)C)ccc(n1)C', 
                      'c1(C[n+]2c(n(cc2)C)cccc1)C[n+]1cn(c(n(C)c1C)C)C', 
                      'C(CC[n+]1cccnN(C)c1)CC', 
                      'c1cccn([n+]2cn(C)c(C)n2C)ccc(C)cc1', 
                      'Cc1[n+](CCCCc2)cc(n(C)c2)C)c(C)=C1', 
                      'c1cc(cc(C)n(c1)C)[n+]1Cc(n(C)cc1)C', 
                      '[n+]1(Cc(n(C)cc1)Cccc1)CCCC[n+]1cn(c(C)C)c1C']
    
    fcd_score = FCD_score(original_data, generated_data)
    print('FCD score:', fcd_score)
    
    # calculate the metrics
    print('Validity rate:', validity_rate(generated_data))
    print('Uniqueness rate:', uniqueness_rate(generated_data))
    print('Reconstructability rate:', reconstructability_rate(generated_data, original_data))
    print('Novelty rate:', novelty_rate(generated_data, original_data))
    print('IntDiv:', IntDiv(generated_data))
    print('KL-divergence:', KL_divergence(np.array(original_data), np.array(generated_data)))
    

