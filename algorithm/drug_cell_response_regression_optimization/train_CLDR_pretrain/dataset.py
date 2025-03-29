import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
import os

# 导入utils.smile_to_graph中的函数
from utils.smile_to_graph import type_check_num_atoms, construct_atomic_number_array, construct_discrete_edge_matrix

# 预处理函数
def preprocess_molecule(smiles, max_node_num=100):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # 检查分子中原子数量是否超过最大节点数
    type_check_num_atoms(mol, max_node_num)

    # 获取原子特征和边特征
    atom_features = construct_atomic_number_array(mol)
    edge_features = construct_discrete_edge_matrix(mol)

    # 将原子特征和边特征转换为所需的格式
    GDSCv2_atomic_num_list = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    atom_features_processed = np.zeros((max_node_num, len(GDSCv2_atomic_num_list)), dtype=np.float32)

    for i, atom_num in enumerate(atom_features):
        if i >= max_node_num:
            break
        ind = GDSCv2_atomic_num_list.index(atom_num) if atom_num in GDSCv2_atomic_num_list else -1
        if ind != -1:
            atom_features_processed[i, ind] = 1.0

    edge_features = torch.tensor(edge_features.argmax(axis=0))      # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
    # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
    edge_features = torch.where(edge_features == 3, 0, edge_features + 1).to(torch.float32)
    
    # 填充边特征矩阵
    if edge_features.shape[0] < max_node_num or edge_features.shape[1] < max_node_num:
        edge_features_processed = F.pad(torch.tensor(edge_features), (0, max_node_num - edge_features.shape[0], 0, max_node_num - edge_features.shape[1]), mode='constant', value=0)

    
    return atom_features_processed, edge_features_processed

# 自定义数据集类
class DrugDataset(InMemoryDataset):
    def __init__(self, root, df, transform=None, pre_transform=None):
        self.df = df
        super(DrugDataset, self).__init__(root, transform, pre_transform)
        self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        if os.path.exists(self.processed_paths[0]):
            print(f"Processed data already exists at {self.processed_paths[0]}. Loading data from {self.processed_paths[0]}.")
            data, slices = torch.load(self.processed_paths[0])
            self.data, self.slices = data, slices
            return

        data_list = []
        from tqdm import tqdm
        for idx, row in tqdm(self.df.iterrows()):
            smiles = row['smiles']
            cell_name = row['cell_name']
            ic50 = row['ic50']

            atom_features, edge_features = preprocess_molecule(smiles)

            data = Data(
                x=torch.tensor(atom_features, dtype=torch.float).unsqueeze(0),
                edge_index=torch.tensor(edge_features, dtype=torch.long).unsqueeze(0).permute(1,2,0),
                cell_name=torch.tensor([cell_name], dtype=torch.long),
                ic50=torch.tensor([ic50], dtype=torch.float)
            )

            data_list.append(data)

        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])
        self.data, self.slices = data, slices

    def len(self):
        return len(self.df)

    # def get(self, idx):
    #     data = self.data.__class__()
    #     if self.slices is None:
    #         raise RuntimeError("Data has not been sliced yet.")
    #     for key in self.data.keys:
    #         item, slices = self.data[key], self.slices[key]
    #         s = list(slice(slices[idx], slices[idx + 1]))
    #         data[key] = item[s] if item.dim() == 0 else item[s[0]]
    #     return data
