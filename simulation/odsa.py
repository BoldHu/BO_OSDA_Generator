import csv

import maze
import os
from ase.calculators import vasp

from maze import Zeolite
from ase.calculators.lj import LennardJones

from rdkit.Chem import AllChem
from ase.optimize import BFGS
import numpy as np
from ase import Atoms
import pandas as pd
from rdkit import Chem
from function_list import *

'-----------------------------得到ODSAs分子的SMILES列表----------------------------'
# folder_path = 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\data'
# guest_smile_csv = pd.read_csv('C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\guests.csv')
folder_path = './data'
guest_smile_csv = pd.read_csv('guests.csv')
guest_smile = np.array(guest_smile_csv['Guest SMILES'])   #ODSAs分子的列表

'-----------------------------rdkit分子化学库遍历计算SMILES三维结构保存为pdb数据文件-----------------------------'
# 现在得到在smiles_pdb文件目录下所有smiles的的三维结构信息，并保存为pdb文件
for i in range(len(guest_smile)):
     try:
          mol = calculate_smiles_structure(guest_smile[i])
     except ValueError as e:
        print(f"Error processing SMILES {guest_smile[i]}: {e}")

'-----------------------------------解析PDB文件以提取原子信息----------------------------------'
# smiles_pdb_folder = 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\smiles_pdb\\' #自定义的文件夹
# directory = 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\atom_path\\'   #自定义的文件夹
smiles_pdb_folder = './smiles_pdb/'
directory = './atom_path/'
for filename in os.listdir(smiles_pdb_folder):
    file_path = os.path.join(smiles_pdb_folder, filename)
#     print(file_path)
    position, atom_list = read_pdb_coordinates(file_path)
    atom_filename = directory + atom_list + '.csv'
    write_coordinates_to_csv(position, atom_filename)
    try:
          co = Atoms(atom_list, positions=[np.array(i) for i in position])
    except ValueError as e:
          print(f"Error processing file {file_path}: {e}")

'-----------------------------------筛选数据-----------------------------'
# binding = pd.read_csv('C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\binding.csv')
binding = pd.read_csv('binding.csv')
binding_host = binding['Host'].values
unique_host = np.unique(binding_host)

df = pd.DataFrame(columns=['smiles', 'Host', 'metric','binding_energy'])
'--------------------------------遍历文件夹准备将SMILES的三维结构和分子筛的三维结构复合---------------------'
for filename in os.listdir(folder_path):
     # bea_zeolite = Zeolite.make(filename.split('.')[0])
     if filename.split('.')[0] in unique_host:
          bea_zeolite = Zeolite.make(filename.split('.')[0])
          bea_atom = Atoms(bea_zeolite.symbols, positions=bea_zeolite.positions) #分子筛的三维结构
          # for atom_pdb in os.listdir('C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\atom_path\\'):
          #      atom_data = os.path.join('C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\atom_path\\', atom_pdb)
          for atom_pdb in os.listdir('./atom_path/'):
               atom_data = os.path.join('./atom_path/', atom_pdb)
               position = pd.read_csv(atom_data).values
               atom_name = atom_pdb.split('.')[0]
               co = Atoms(atom_name, positions=[np.array(i) for i in position])    #SMILES的三维结构

               complex_atom,odsa_smiles = Zeolite.integrate_adsorbate(bea_zeolite,co) #复合结构
               feishi = Atoms(complex_atom.symbols, positions = complex_atom.positions,cell=complex_atom.cell) #复合三维结构

               '----------------------利用ase库计算结构能量,优化结构------------------------'
               optimizer = BFGS(feishi)
               feishi.calc = LennardJones(sigma=3.41, epsilon=1.67)
               optimizer.run(fmax=0.01,steps=2000)


               optimizer = BFGS(bea_atom)
               bea_atom.calc = LennardJones(sigma=1.4, epsilon=0.67)
               optimizer.run(fmax=0.01,steps=1500)

               optimizer = BFGS(odsa_smiles)
               odsa_smiles.calc = LennardJones(sigma=1, epsilon=0.67)
               optimizer.run(fmax=0.01,steps=200)

               energy_all = feishi.get_potential_energy()
               energy_bea = bea_atom.get_potential_energy()
               energy_odsa = odsa_smiles.get_potential_energy()

               c = energy_all - energy_odsa - energy_bea
               df = df._append({'smiles':atom_name,'Host':filename.split('.')[0],'metric':'bfgs','binding_energy':c}, ignore_index=True)  #这一行会有pandas版本问题
     else:
          continue

df.to_csv('value.csv', index=False)