import csv
from ase import Atoms
from rdkit.Chem import AllChem
from rdkit import Chem
import os
import numpy as np

def calculate_smiles_structure(smiles):
    # smiles_pdb_path = 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\smiles_pdb\\'
    smiles_pdb_path = './smiles_pdb/'
    mol = Chem.MolFromSmiles(smiles)
    # 检查分子是否为空
    if mol is None:
        print("无法解析SMILES字符串")
    else:
        # 生成三维结构
        # 注意：这里使用UFF力场进行初步的结构优化
        # 你也可以尝试使用其他力场，如MMFF94s，但需要额外安装和配置
        # AllChem.EmbedMolecule(mol)
        params = AllChem.ETKDG()
        if AllChem.EmbedMolecule(mol, params) == -1:
            raise ValueError(f"Could not embed molecule for SMILES: {smiles}")
        AllChem.MMFFOptimizeMolecule(mol)
        # 保存为PDB文件
        pdb_filename = smiles_pdb_path + smiles + '.pdb'
        Chem.MolToPDBFile(mol, pdb_filename)
    return mol

def calculate_smiles_structure_with_hydrogen(smiles):
    # smiles_pdb_path = 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\smiles_pdb\\'
    smiles_pdb_path = './smiles_pdb/'
    mol = Chem.MolFromSmiles(smiles)
    # 检查分子是否为空
    if mol is None:
        print("无法解析SMILES字符串")
    else:
        # 生成三维结构
        # 注意：这里使用UFF力场进行初步的结构优化
        # 你也可以尝试使用其他力场，如MMFF94s，但需要额外安装和配置
        # AllChem.EmbedMolecule(mol)
        # 添加氢原子
        mol = Chem.AddHs(mol) 
        params = AllChem.ETKDG()
        if AllChem.EmbedMolecule(mol, params) == -1:
            raise ValueError(f"Could not embed molecule for SMILES: {smiles}")
        AllChem.MMFFOptimizeMolecule(mol)
        # 保存为PDB文件
        pdb_filename = smiles_pdb_path + smiles + '_H.pdb'
        Chem.MolToPDBFile(mol, pdb_filename)
    return mol

def read_pdb_coordinates(pdb_file):
    atom_start_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    element_types = []
    coordinates = []
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith(('ATOM', 'HETATM')):
                for i in range(len(line) - 1, -1, -1):
                    if line[i] in atom_start_list:
                        element_type = line[i]
                        break
                element_types.append(element_type)
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coordinates.append([x, y, z])
        aotm_list = ''.join(element_types)
    return coordinates, aotm_list

def write_coordinates_to_csv(coordinates, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入标题行（可选）
        writer.writerow(['x', 'y', 'z'])
        # 写入坐标数据
        for coord in coordinates:
            writer.writerow(coord)
            
if __name__ == '__main__':
    # 计算SMILES字符串对应的分子结构
    smiles = 'N[C@]12C[C@H]3C[C@H](C[C@H](C3)C1)C2'
    # mol = calculate_smiles_structure(smiles)
    # # 读取PDB文件中的坐标数据
    # pdb_file = './smiles_pdb/N[C@]12C[C@H]3C[C@H](C[C@H](C3)C1)C2'
    # position, atom_list = read_pdb_coordinates(pdb_file)
    # # 将坐标数据写入CSV文件
    # csv_filename = './atom_path/N[C@]12C[C@H]3C[C@H](C[C@H](C3)C1)C2.csv'
    # write_coordinates_to_csv(position, csv_filename)
    # print('原子类型：', atom_list)
    # print('坐标数据已保存到：', csv_filename)
    # co = Atoms(atom_list, positions=[np.array(i) for i in position])
    
    mol = calculate_smiles_structure_with_hydrogen(smiles)
    # 读取PDB文件中的坐标数据
    pdb_file = './smiles_pdb/N[C@]12C[C@H]3C[C@H](C[C@H](C3)C1)C2_H.pdb'
    position, atom_list = read_pdb_coordinates(pdb_file)
    # 将坐标数据写入CSV文件
    csv_filename = './atom_path/N[C@]12C[C@H]3C[C@H](C[C@H](C3)C1)C_H.csv'
    write_coordinates_to_csv(position, csv_filename)
    print('原子类型：', atom_list)
    print('坐标数据已保存到：', csv_filename)
    co = Atoms(atom_list, positions=[np.array(i) for i in position])
