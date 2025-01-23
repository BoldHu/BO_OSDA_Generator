# convert the smiles to canonical smiles
from rdkit import Chem 

# origin_smile = 'Cn1cc[n+](Cc2ccccc2C[n+]2ccn(C)c2C)c1C'
# canonical_smile = Chem.MolToSmiles(Chem.MolFromSmiles(origin_smile))
# real_cano_smile = 'Cc1n(C)cc[n+]1Cc1ccccc1C[n+]1ccn(C)c1C'
# print('origin smile is :', origin_smile)
# print('canonical smile is :', canonical_smile)
# print('real canonical smile is :', real_cano_smile)

# # random the canonical smiles
# import random
# random_smiles_list = []
# for i in range(10):
#     # rebuild the smiles
#     canonical_smile = 'Cc1n(C)cc[n+]1Cc1ccccc1C[n+]1ccn(C)c1C'
#     canonical_mol = Chem.MolFromSmiles(canonical_smile)
#     random_smiles = Chem.MolToSmiles(canonical_mol, doRandom=True)
#     random_smiles_list.append(random_smiles)

# print('random smiles list is :', random_smiles_list)

# # draw the molecule to figures/moleculars folder
# from rdkit.Chem import Draw
# from rdkit.Chem import AllChem
# import os

# # draw the canonical smiles
# canonical_mol = Chem.MolFromSmiles(real_cano_smile)
# img = Draw.MolToImage(canonical_mol)
# img.save('figures/moleculars/canonical_{}.png'.format(real_cano_smile))
# # draw the random smiles and name it as smiles value
# for i in range(10):
#     random_mol = Chem.MolFromSmiles(random_smiles_list[i])
#     img = Draw.MolToImage(random_mol)
#     img.save('figures/moleculars/{}.png'.format(random_smiles_list[i]))

origin_smile = 'C[N+](C)(C)C12CC3CC(CC(C3)C1)C2'
canonical_smile = Chem.MolToSmiles(Chem.MolFromSmiles(origin_smile))
real_cano_smile = 'C[N+](C)(C)C12CC3CC(CC(C3)C1)C2'
print('origin smile is :', origin_smile)
print('canonical smile is :', canonical_smile)
print('real canonical smile is :', real_cano_smile)

# random the canonical smiles
import random
random_smiles_list = []
for i in range(10):
    # rebuild the smiles
    canonical_smile = 'C[N+](C)(C)C12CC3CC(CC(C3)C1)C2'
    canonical_mol = Chem.MolFromSmiles(canonical_smile)
    random_smiles = Chem.MolToSmiles(canonical_mol, doRandom=True)
    random_smiles_list.append(random_smiles)

print('random smiles list is :', random_smiles_list)

# draw the molecule to figures/moleculars folder
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os

# draw the canonical smiles
canonical_mol = Chem.MolFromSmiles(real_cano_smile)
img = Draw.MolToImage(canonical_mol)
img.save('figures/positive/canonical_{}.png'.format(real_cano_smile))
# draw the random smiles and name it as smiles value
for i in range(10):
    random_mol = Chem.MolFromSmiles(random_smiles_list[i])
    img = Draw.MolToImage(random_mol)
    img.save('figures/positive/{}.png'.format(random_smiles_list[i]))

# negative_smiles = ['Fc1cnc(NC(=O)[C@H](CC2CCOCC2)c3ccc(cc3)S(=O)(=O)C4CC4)s1', 'Oc1ccc(cc1)\N=C(\Cc2ccc(Cl)cc2)/c3ccc(O)c(O)c3O', 
# 'COc1cc(ccc1O)C(O)C(COC(=O)\C=C\c2ccc(O)cc2)Oc3c(OC)cc(\C=C\COC(=O)\C=C\c4ccc(O)cc4)cc3OC',
# 'CC(=CCC\C(=C\Cc1c(O)c2C(=O)C3=C(Oc2c4C=CC(C)(C)Oc14)c5c(O)cc(O)c6OC(C)(C)C(C3)c56)\C)C', 
# 'CC(C)(C)OC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc2ccccc2)NC(=O)[C@@H]3CCCN3C(=O)[C@@H](N)Cc4ccc(O)cc4',
# 'COc1ccccc1OCCN2CCN(CC2)C3=NN(CN4N=C(C=CC4=O)N5CCN(CCOc6ccccc6OC)CC5)C(=O)C=C3',
# 'CCCCCCCCCCCC[C@@H](O)[C@H]1CC[C@@H](O1)[C@H](O)CCC(O)CCCC(O)CCC[C@@H](O)CC2=C[C@H](C)OC2=O']

negative_smiles = ['Fc1cnc(NC(=O)[C@H](CC2CCOCC2)c3ccc(cc3)S(=O)(=O)C4CC4)s1',
'COc1cc(ccc1O)C(O)C(COC(=O)\C=C\c2ccc(O)cc2)Oc3c(OC)cc(\C=C\COC(=O)\C=C\c4ccc(O)cc4)cc3OC',
'CC(=CCC\C(=C\Cc1c(O)c2C(=O)C3=C(Oc2c4C=CC(C)(C)Oc14)c5c(O)cc(O)c6OC(C)(C)C(C3)c56)\C)C',
'CC(C)(C)OC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc2ccccc2)NC(=O)[C@@H]3CCCN3C(=O)[C@@H](N)Cc4ccc(O)cc4',
'COc1ccccc1OCCN2CCN(CC2)C3=NN(CN4N=C(C=CC4=O)N5CCN(CCOc6ccccc6OC)CC5)C(=O)C=C3',
'CCCCCCCCCCCC[C@@H](O)[C@H]1CC[C@@H](O1)[C@H](O)CCC(O)CCCC(O)CCC[C@@H](O)CC2=C[C@H](C)OC2=O']

# draw the negative smiles
for i in range(len(negative_smiles)):
    negative_mol = Chem.MolFromSmiles(negative_smiles[i])
    # convert the negative smiles to image with transparent background
    img = Draw.MolToImage(negative_mol, kekulize=True, wedgeBonds=True, imageType='png', size=(300, 300), transparent=True)
    img.save('figures/negative/{}.png'.format(negative_smiles[i]))
    
mole = 'CC(C)NC(C)C'
# draw the mole
mole_mol = Chem.MolFromSmiles(mole)
img = Draw.MolToImage(mole_mol)
img.save('figures/moleculars/{}.png'.format(mole))
    

