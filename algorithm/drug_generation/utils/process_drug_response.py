import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math
import pandas as pd
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
import random
import pickle
import sys
import matplotlib.pyplot as plt
import argparse
import torch,pdb
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from torch_geometric.data import Batch
from torch_geometric.data import Data
# from GDSC2_utils import GDSC2Dataset

def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True

"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""

folder = "data/"
#folder = ""


allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

"""
The following code will convert the SMILES format into onehot format
"""

def atom_features(atom):
    # print(atom.GetChiralTag())
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding(atom.GetChiralTag(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()]+[atom.GetAtomicNum()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
    
    
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    # Chem.MolFromSmiles('CC[C@H](F)Cl', useChirality=True)
    # mol = Chem.MolFromSmiles(smile, useChirality=True, isomericSmiles= True, kekuleSmiles = True)
    # pdb.set_trace()
    c_size = mol.GetNumAtoms()
    
    features = []
    node_dict = {}
    
    for atom in mol.GetAtoms():
        node_dict[str(atom.GetIdx())] = atom
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    
    # bonds
    num_bond_features = 5
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # if allowable_features['possible_bond_dirs'].index(bond.GetBondDir()) !=0:
                # pdb.set_trace()
                # print(smile, allowable_features['possible_bond_dirs'].index(bond.GetBondDir()))
            edge_feature = [
                bond.GetBondTypeAsDouble(), 
                bond.GetIsAromatic(),
                # 芳香键
                bond.GetIsConjugated(),
                # 是否为共轭键
                bond.IsInRing(),             
                allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    
    return c_size, features, edge_index, edge_attr

def load_drug_smile():
    reader = csv.reader(open(folder+"GDSCv2/drug_smiles.csv"))
    next(reader, None)

    smile_graph = {}
    drug_dict = {}
    
    for item in reader:
        name = item[0]
        smile = item[3]

        if name in drug_dict:
            continue
        else:
            g = smile_to_graph(smile)
            smile_graph[smile] = g
            drug_dict[name] = smile

    return drug_dict, smile_graph

def save_cell_mut_matrix(cell_csv_path):
    f = open(cell_csv_path)
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        cell_id = item[1]
        mut = item[5]
        is_mutated = int(item[6])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))
    
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    return cell_dict, cell_feature


"""
This part is used to extract the drug - cell interaction strength. it contains IC50, AUC, Max conc, RMSE, Z_score
"""
def save_mix_drug_cell_matrix():
    from tqdm import tqdm
    f = open(folder+"GDSCv2/PANCANCER_IC.csv")
    
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_smiles, smile_graph = load_drug_smile()

    temp_data = []
    cell_lines_map= {}
   

    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    data_dict = {}
    
    xd = []
    xc = []
    y = []

    random.shuffle(temp_data)
    
    for data in temp_data:
        drug, cell, ic50 = data
        
        if drug in drug_smiles.keys() and cell in cell_dict:
              
            if data_dict.get(drug):  
                data_dict[drug]['xd'].append(drug_smiles[drug])
                data_dict[drug]['xc'].append(cell_feature[cell_dict[cell]])
                data_dict[drug]['y'].append(ic50)
                data_dict[drug]['cell_name'].append(cell)

            else:
                data_dict[drug]= {
                    'xd':[],
                    'xc':[],
                    'y':[],
                    'drug_name':drug,
                    'cell_name':[]
                }
                data_dict[drug]['xd'].append(drug_smiles[drug])
                data_dict[drug]['xc'].append(cell_feature[cell_dict[cell]])
                data_dict[drug]['y'].append(ic50)
                data_dict[drug]['cell_name'].append(cell)
        

    
    for drugName in tqdm(data_dict):
        data_dict[drugName]['xd'] = np.asarray(data_dict[drugName]['xd'])
        data_dict[drugName]['xc'] = np.asarray(data_dict[drugName]['xc'])
        data_dict[drugName]['y'] = np.asarray(data_dict[drugName]['y'])
        data_dict[drugName]['cell_name'] = np.asarray(data_dict[drugName]['cell_name'])
        data_dict[drugName]['drug_num'] = data_dict[drugName]['xd'].shape[0]
    

    
    # data_dict = np.load("/home/lk/project/117graphDRP/transedrp/data/NCI-60/process_temp/data_dict.npy",allow_pickle=True).item()
    # np.save("/home/lk/project/117graphDRP/transedrp/data/GDSCv2/data_dict.npy",data_dict)
    
    
    # _ = GDSC2Dataset(root='data', dataset = 'GDSC2', rawdata=data_dict, smile_graph=smile_graph)


def seed_torch(seed=532):
    seed = int(seed)
    random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    parser.add_argument('--choice', type=int, required=False, default=0, help='0.mix test, 1.saliency value, 2.drug blind, 3.cell blind')
    args = parser.parse_args()
    choice = args.choice
    seed_torch()

    save_mix_drug_cell_matrix()
