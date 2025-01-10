# convert the smiles to canonical smiles
from rdkit import Chem 

origin_smile = 'Cn1cc[n+](Cc2ccccc2C[n+]2ccn(C)c2C)c1C'
canonical_smile = Chem.MolToSmiles(Chem.MolFromSmiles(origin_smile))
real_cano_smile = 'Cc1n(C)cc[n+]1Cc1ccccc1C[n+]1ccn(C)c1C'
print('origin smile is :', origin_smile)
print('canonical smile is :', canonical_smile)
print('real canonical smile is :', real_cano_smile)