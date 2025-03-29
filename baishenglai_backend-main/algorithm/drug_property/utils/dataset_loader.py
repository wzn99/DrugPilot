import torch
from torch_geometric.data import Data
from rdkit import Chem
from torch.utils.data import Dataset
import pandas as pd
import networkx as nx
import numpy as np
import pdb

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def mol_to_graph(mol):
    # 原子数
    c_size = mol.GetNumAtoms()

    # 原子特征
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    # 使用键信息构造边缘索引
    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((start, end))
        edge_index.append((end, start))  # 由于图是无向的，添加两个方向

    # 转换为PyTorch张量
    # x = torch.tensor(features, dtype=torch.float)
    # 假设 features 是一个包含多个 numpy.ndarrays 的列表
    features_np = np.array(features)  # 将列表转换为单个 NumPy 数组
    x = torch.from_numpy(features_np).float()  # 将 NumPy 数组转换为 PyTorch 张量

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # pdb.set_trace()
    return c_size, x, edge_index


class MolDataset(Dataset):
    def __init__(self, smiles, labels):
        self.smiles = smiles
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        label = self.labels[idx]

        mol = Chem.MolFromSmiles(smile)

        _, x, edge_index = mol_to_graph(mol)
        # pdb.set_trace()
        # 为PyTorch Geometric创建Data实例
        data = Data(x=x,
                    edge_index=edge_index,
                    y=torch.tensor([label], dtype=torch.float),
                    smile=smile)

        return data

def load_data(dataset_path):

    df = pd.read_csv(dataset_path)

    smiles_list = df['smiles'].values.tolist()

    # The labels start from the 2nd column onwards
    labels = df.iloc[:, 1:].values

    # Replace 'void' values with a specific marker, e.g., inf
    labels = np.where(pd.isna(labels), float('inf'), labels).astype(float)
    dataset = MolDataset(smiles_list, labels)

    return dataset
