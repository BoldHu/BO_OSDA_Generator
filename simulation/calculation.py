import csv
import os
from ase.calculators import vasp

from maze import Zeolite
from ase.calculators.lj import LennardJones

from rdkit.Chem import AllChem
from ase.optimize import BFGS, LBFGS
import numpy as np
from ase import Atoms
import pandas as pd
from rdkit import Chem
from function_list import *
from download_zeolite_cif import download_zeolite_files

def calculate_binding_energy(osda_smiles, zeolite_name, zeolite_dir = './simulation/zeolite_files/', atom_dir = './simulation/atom_path/', smiles_dir = './simulation/smiles_pdb/'):
    """
    Calculate the binding energy of the specified OSDA with the specified zeolite.

    Parameters:
    osda_smiles (str): SMILES string of the OSDA.
    zeolite_name (str): Name of the zeolite.
    data_dir (str): Directory where the CIF files are stored.

    Returns:
    float: The binding energy of the OSDA with the zeolite.
    """
    
    # get the paths for the zeolite CIF file, atom CSV file, and SMILES PDB file
    zeolite_cif_path = os.path.join(zeolite_dir, f"{zeolite_name}.cif")
    smiles_pdb_path = os.path.join(smiles_dir, f"{osda_smiles}.pdb")
    
    # check if the zeolite.cif is in the directory
    if not os.path.exists(zeolite_cif_path):
        download_zeolite_files([zeolite_name], zeolite_dir)
    
    # calculate the smiles structure of the OSDA
    try:
          mol = calculate_smiles_structure(osda_smiles)
    except ValueError as e:
        print(f"Error processing SMILES {osda_smiles}: {e}")
        # return -inf
        return -float('inf')
    
    # read the pdf file to get the atom list
    position, atom_list = read_pdb_coordinates(smiles_pdb_path)
    atom_save_path = os.path.join(atom_dir, f"{atom_list}.csv")
    write_coordinates_to_csv(position, atom_save_path)
    try:
        co = Atoms(atom_list, positions=[np.array(i) for i in position])
    except ValueError as e:
        print(f"Error processing file {smiles_pdb_path}: {e}")
    
    # get the zeolite object and osda atom object
    bea_zeolite = Zeolite.make(iza_code=zeolite_name, data_dir='./simulation/zeolite_files/')
    zeolite_atom = Atoms(bea_zeolite.symbols, positions=bea_zeolite.positions)
    
    # integrate the adsorbate with the zeolite
    complex_atom, odsa_smiles = Zeolite.integrate_adsorbate(bea_zeolite, co)
    feishi = Atoms(complex_atom.symbols, positions = complex_atom.positions,cell=complex_atom.cell)
    
    # optimize the structures
    feishi.calc = LennardJones(sigma=3.41, epsilon=1.67)    
    optimizer = BFGS(feishi)
    optimizer.run(fmax=0.01,steps=2000)
    
    zeolite_atom.calc = LennardJones(sigma=1.4, epsilon=0.67)
    optimizer = BFGS(zeolite_atom)
    optimizer.run(fmax=0.01,steps=1500)

    odsa_smiles.calc = LennardJones(sigma=1, epsilon=0.67)
    optimizer = BFGS(odsa_smiles)
    optimizer.run(fmax=0.01,steps=200)
    
    energy_all = feishi.get_potential_energy()
    energy_bea = zeolite_atom.get_potential_energy()
    energy_odsa = odsa_smiles.get_potential_energy()
    
    c = energy_all - energy_odsa - energy_bea
    
    return odsa_smiles, zeolite_name, 'bfgs', c

def calculate_binding_energy_l(osda_smiles, zeolite_name, zeolite_dir = './simulation/zeolite_files/', atom_dir = './simulation/atom_path/', smiles_dir = './simulation/smiles_pdb/'):
    """
    Calculate the binding energy of the specified OSDA with the specified zeolite.

    Parameters:
    osda_smiles (str): SMILES string of the OSDA.
    zeolite_name (str): Name of the zeolite.
    data_dir (str): Directory where the CIF files are stored.

    Returns:
    float: The binding energy of the OSDA with the zeolite.
    """
    
    # get the paths for the zeolite CIF file, atom CSV file, and SMILES PDB file
    zeolite_cif_path = os.path.join(zeolite_dir, f"{zeolite_name}.cif")
    smiles_pdb_path = os.path.join(smiles_dir, f"{osda_smiles}_H.pdb")
    
    # check if the zeolite.cif is in the directory
    if not os.path.exists(zeolite_cif_path):
        download_zeolite_files([zeolite_name], zeolite_dir)
    
    # calculate the smiles structure of the OSDA
    try:
          mol = calculate_smiles_structure_with_hydrogen(osda_smiles)
    except ValueError as e:
        print(f"Error processing SMILES {osda_smiles}: {e}")
        # return -inf
        return -float('inf')
    
    # read the pdf file to get the atom list
    position, atom_list = read_pdb_coordinates(smiles_pdb_path)
    atom_save_path = os.path.join(atom_dir, f"{atom_list}_H.csv")
    write_coordinates_to_csv(position, atom_save_path)
    try:
        co = Atoms(atom_list, positions=[np.array(i) for i in position])
    except ValueError as e:
        print(f"Error processing file {smiles_pdb_path}: {e}")
    
    # get the zeolite object and osda atom object
    bea_zeolite = Zeolite.make(iza_code=zeolite_name, data_dir='./simulation/zeolite_files/')
    zeolite_atom = Atoms(bea_zeolite.symbols, positions=bea_zeolite.positions)
    
    # integrate the adsorbate with the zeolite
    complex_atom, odsa_smiles = Zeolite.integrate_adsorbate(bea_zeolite, co)
    feishi = Atoms(complex_atom.symbols, positions = complex_atom.positions,cell=complex_atom.cell)
    
    # optimize the structures
    feishi.calc = LennardJones(sigma=3.41, epsilon=1.67)
    optimizer = LBFGS(feishi)
    optimizer.run(fmax=0.01,steps=2000)
    
    zeolite_atom.calc = LennardJones(sigma=1.4, epsilon=0.67)   
    optimizer = LBFGS(zeolite_atom)
    optimizer.run(fmax=0.01,steps=1500)

    odsa_smiles.calc = LennardJones(sigma=1, epsilon=0.67)
    optimizer = LBFGS(odsa_smiles)
    optimizer.run(fmax=0.01,steps=200)
    
    energy_all = feishi.get_potential_energy()
    energy_bea = zeolite_atom.get_potential_energy()
    energy_odsa = odsa_smiles.get_potential_energy()
    
    c = energy_all - energy_odsa - energy_bea
    
    return odsa_smiles, zeolite_name, 'bfgs', c

def calculate_binding_energy_H(osda_smiles, zeolite_name, zeolite_dir = './simulation/zeolite_files/', atom_dir = './simulation/atom_path/', smiles_dir = './simulation/smiles_pdb/'):
    """
    Calculate the binding energy of the specified OSDA with the specified zeolite with high accuracy using calculate_smiles_structure_with_hydrogen

    Parameters:
    osda_smiles (str): SMILES string of the OSDA.
    zeolite_name (str): Name of the zeolite.
    data_dir (str): Directory where the CIF files are stored.

    Returns:
    float: The binding energy of the OSDA with the zeolite.
    """
    
    # get the paths for the zeolite CIF file, atom CSV file, and SMILES PDB file
    zeolite_cif_path = os.path.join(zeolite_dir, f"{zeolite_name}.cif")
    smiles_pdb_path = os.path.join(smiles_dir, f"{osda_smiles}_H.pdb")
    
    # check if the zeolite.cif is in the directory
    if not os.path.exists(zeolite_cif_path):
        download_zeolite_files([zeolite_name], zeolite_dir)
    
    # calculate the smiles structure of the OSDA
    try:
          mol = calculate_smiles_structure_with_hydrogen(osda_smiles)
    except ValueError as e:
        print(f"Error processing SMILES {osda_smiles}: {e}")
        # return -inf
        return -float('inf')
    
    # read the pdf file to get the atom list
    position, atom_list = read_pdb_coordinates(smiles_pdb_path)
    atom_save_path = os.path.join(atom_dir, f"{atom_list}_H.csv")
    write_coordinates_to_csv(position, atom_save_path)
    try:
        co = Atoms(atom_list, positions=[np.array(i) for i in position])
    except ValueError as e:
        print(f"Error processing file {smiles_pdb_path}: {e}")
    
    # get the zeolite object and osda atom object
    bea_zeolite = Zeolite.make(iza_code=zeolite_name, data_dir='./simulation/zeolite_files/')
    zeolite_atom = Atoms(bea_zeolite.symbols, positions=bea_zeolite.positions)
    
    # integrate the adsorbate with the zeolite
    complex_atom, odsa_smiles = Zeolite.integrate_adsorbate(bea_zeolite, co)
    feishi = Atoms(complex_atom.symbols, positions = complex_atom.positions,cell=complex_atom.cell)
    
    # optimize the structures
    feishi.calc = LennardJones(sigma=3.41, epsilon=1.67)    
    optimizer = BFGS(feishi)
    optimizer.run(fmax=0.01,steps=2000)
    
    zeolite_atom.calc = LennardJones(sigma=1.4, epsilon=0.67)
    optimizer = BFGS(zeolite_atom)
    optimizer.run(fmax=0.01,steps=2000)

    odsa_smiles.calc = LennardJones(sigma=1, epsilon=0.67)
    optimizer = BFGS(odsa_smiles)
    optimizer.run(fmax=0.01,steps=2000)
    
    energy_all = feishi.get_potential_energy()
    energy_bea = zeolite_atom.get_potential_energy()
    energy_odsa = odsa_smiles.get_potential_energy()
    
    c = energy_all - energy_odsa - energy_bea
    
    return odsa_smiles, zeolite_name, 'bfgs', c

if __name__ == '__main__':
    # Example usage
    osda_smiles = 'CCNCC'
    zeolite_name = 'ACO'
    binding_energy = calculate_binding_energy(osda_smiles, zeolite_name)
    print(f"Binding energy of {osda_smiles} with {zeolite_name}: {binding_energy}")
    
    binding_energy_H = calculate_binding_energy_H(osda_smiles, zeolite_name)
    print(f"Binding energy of {osda_smiles} with {zeolite_name}: {binding_energy_H}")
    
    binding_energy_l = calculate_binding_energy_l(osda_smiles, zeolite_name)
    print(f"Binding energy of {osda_smiles} with {zeolite_name}: {binding_energy_l}")
    
    # save the binding energy to a CSV file
    with open('./simulation/results/binding_energy.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['OSDA', 'Zeolite', 'Method', 'Binding Energy'])
        if binding_energy != -float('inf'):
            writer.writerow(binding_energy)
        if binding_energy_H != -float('inf'):
            writer.writerow(binding_energy_H)
        if binding_energy_l != -float('inf'):
            writer.writerow(binding_energy_l)