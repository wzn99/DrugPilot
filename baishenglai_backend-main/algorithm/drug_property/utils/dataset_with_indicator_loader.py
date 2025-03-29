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

    # # 键特征
    # edges = []
    # for bond in mol.GetBonds():
    #     edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    # g = nx.Graph(edges).to_directed()
    # edge_index = []
    # for e1, e2 in g.edges:
    #     edge_index.append([e1, e2])

    # return c_size, features, edge_index
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
    def __init__(self, smiles, labels, indicators, task_output_dims):
        self.smiles = smiles
        self.labels = labels  # This should now be a list of lists or a 2D array where each row is a molecule and each column a task
        self.indicators = indicators  # Similar structure to labels
        self.task_output_dims = task_output_dims  # Number of labels per task
        
    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        label = self.labels[idx]
        indicator = self.indicators[idx]

        mol = Chem.MolFromSmiles(smile)

        _, x, edge_index = mol_to_graph(mol)
        # Prepare labels as a list of tensors, each corresponding to a task; same for indicator
        y = []
        ind = []
        start_idx = 0
        for output_dim in self.task_output_dims:
            end_idx = start_idx + output_dim
            task_labels = label[start_idx:end_idx]
            task_ind = indicator[start_idx:end_idx]
            y.append(torch.tensor([task_labels], dtype=torch.float))
            ind.append(torch.tensor([task_ind], dtype=torch.float))
            start_idx = end_idx
        # pdb.set_trace()

        data = Data(x=x,
                    edge_index=edge_index,
                    y=y,  # Now y is a list of tensors
                    indicator=ind,
                    smile=smile)

        return data

def load_data(config):

    df = pd.read_csv(config['dataset_path'])

    smiles_list = df['smiles'].values.tolist()

    # Assuming the labels start from the 2nd column onwards
    labels = df.iloc[:, 1:].values
    
    df_ind = pd.read_csv(config['indicator_path'])
    
    indicators = df_ind.iloc[:, 1:].values
    
    dataset = MolDataset(smiles_list, labels, indicators, config['task_output_dims'])

    return dataset
