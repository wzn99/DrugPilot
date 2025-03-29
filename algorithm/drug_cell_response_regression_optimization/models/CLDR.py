import sys

sys.path.insert(0,"./")
sys.path.insert(0,"./models")


import math
import pdb
import re
import os
import torch.nn.functional as F

import random
import datetime
from utils import *

# from models.model_graphdrp import GraphDRP
# from models.model_graphdrp_reg_num2 import GraphDRP
from model_transedrp_reg_KGE import TransEDRP
# from models.model_transedrp_reg_num import TransEDRP

import argparse
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import torch.nn as nn
import torch
from tqdm import tqdm
from random import shuffle
import pandas as pd
import numpy as np
import clip
from copy import deepcopy
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset

# training function at each epoch
# training function at each epoch
import torch
import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.device_count()

PRECISION = 3




def generate_samples(model, data, start, end):
    # # data: list, consist of [drug smile, cell line, ic50]
    # descriptions = []
    # assert end - start == int('1'+'0'*PRECISION)
    
    # # if model.training:    
    # # for ic50 in range(start,end,1):
    # for idx, ic50 in enumerate(range(0,int('1'+'0'*PRECISION),1)):
    #     # 
    #     # pdb.set_trace()
    #     # print(ic50)
    #     des = "zero point" + num2english(ic50/int('1'+'0'*PRECISION), PRECISION)
        
    #     descriptions.append(des)
    #         # pdb.set_trace()
    # text = clip.tokenize(descriptions,context_length=100).to(device)
    # # pdb.set_trace()
    # text_features = model.encode_num(text)
    
    return ['text_features']
            
            
def predicting(model, device, loader, loader_type, args):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader)):
            data = data.to(device)
            data_ = deepcopy(data)
            
            fusion_features = model.infer(data)
            
            number_features = generate_samples(model, data, 0, int('1'+'0'*PRECISION))
            # gt_sentence_features = gt_sentence_features/gt_sentence_features.norm(dim=1,keepdim=True)
            
            preds = torch.Tensor()
            for data_item_index in range(len(data)):
                des = "The drug response value between " + data.smiles[data_item_index] + " and "+ data.cell_name[data_item_index] +" is "
                text = clip.tokenize([des,des],context_length=300).to(device)
                
                text_features = model.encode_text(text)[0]
                logits = torch.Tensor()
                for interval_index in range(10):
                    ten_num = int(int('1'+'0'*PRECISION)/10)
                    sentence_features = torch.cat((text_features.repeat(ten_num,1),number_features[interval_index*ten_num:(interval_index+1)*ten_num]),axis=1)
            
                    sentence_features = model.transformer_fusion(sentence_features)
                    sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

                    

                    # pdb.set_trace()
                    logit = model.logit_scale * fusion_features[data_item_index].unsqueeze(0) @ sentence_features.t()
                    logits = torch.cat((logits,logit.cpu()),1)
                
                
                pred = (torch.argmax(logits,1)/int('1'+'0'*PRECISION)).view(-1, 1)
                preds = torch.cat((preds,pred),0)
                
            # pdb.set_trace() 

            total_preds = torch.cat((total_preds, preds), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
        
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


'''freeze'''


def dateStr():
    return (
        str(datetime.datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .split(".")[0]
        .replace("_", "")
    )

import pdb, torch
import numpy as np
from tqdm import tqdm
# from models.gat_gcn_transformer import GAT_GCN_Transformer
from rdkit import Chem
import networkx as nx
import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math

import random

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
    # mol = Chem.MolFromSmiles(smile, isomericSmiles= True), kekuleSmiles = True)
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
            # print(smile,allowable_features['possible_bond_dirs'].index(bond.GetBondDir()))
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

def getTCNNsMatrix(smiles):
    c_chars, _ = charsets(smiles)
    c_length = max(map(len, map(string2smiles_list, smiles)))
    
    smiles = np.array(smiles)
    count = smiles.shape[0]

    smiles = [string2smiles_list(smiles[i]) for i in range(count)]
    
    canonical = smiles_to_onehot(smiles, c_chars, c_length)
    
    save_dict = {}
    save_dict["canonical"] = canonical
    save_dict["c_chars"] = c_chars
    
    return  canonical

def charsets(smiles):
    from functools import reduce
    union = lambda x, y: set(x) | set(y)
    c_chars = list(reduce(union, map(string2smiles_list, smiles)))
    i_chars = list(reduce(union, map(string2smiles_list, smiles)))
    return c_chars, i_chars

def string2smiles_list(string):
    char_list = []
    i = 1
    while i < len(string):
        c = string[i]
        if c.islower():
            char_list.append(string[i-1:i+1])
            i += 1
        else:
            char_list.append(string[i-1])
        i += 1
    if not string[-1].islower():
        char_list.append(string[-1])
    return char_list

def smiles_to_onehot(smiles, c_chars, c_length):
    
    c_ndarray = np.ndarray(shape=(len(smiles), len(c_chars), c_length), dtype=np.float32)
    for i in range(len(smiles)):
        c_ndarray[i, ...] = onehot_encode(c_chars, smiles[i], c_length)
    return c_ndarray

def onehot_encode(char_list, smiles_string, length):
    
    encode_row = lambda char: map(int, [c == char for c in smiles_string])
    
    ans = [ list(x) for x in map(encode_row, char_list)]
    ans = np.array(ans)
    if ans.shape[1] < length:
        
        residual = np.zeros((len(char_list), length - ans.shape[1]), dtype=np.int8)
        ans = np.concatenate((ans, residual), axis=1)
    return ans

def load_cell_mut_matrix(cell_lines_path):
    f = open(cell_lines_path)
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
            
def seed_torch(seed=42):
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
    
def save2Txt(data,save_txt_path):
    data=np.array(data)
    # np.save(save_txt_path,data)   # 保存为.npy格式
    np.savetxt(save_txt_path, data, delimiter=',', fmt='%.2f')
        
# def main(cell_lines_path, drug_smils, model_pth, record_txt_save_path):
def infer(config, yaml_path, infer_model):

    cell_lines_path = "data/PANCANCER_Genetic_feature.csv"
    drug_smiles = "data/to_infer_drug.txt"
    
    cuda_name = 'cuda:2'
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    
    cell_dict, cell_features = load_cell_mut_matrix(cell_lines_path)

    # cell_dict = {value:key for key,value in cell_dict.items()}


    with open(drug_smiles, 'r') as file:  
        # 迭代文件对象，逐行读取内容并存储到列表中  
        drug_list = [line.strip() for line in file]  

    # canonical = getTCNNsMatrix(drug_list)
    
    max_record = []
    
   
    record = []
    
    dift = 0.4224209993928775
    aim_ic50 = 0.4224209993928775
    for i in range(len(drug_list)):
        
            
        smiles = drug_list[i]
        # tCNNs_drug_matrix = canonical[i]
        
        cell_feature = cell_features[cell_dict['687799']]
        
        c_size, features, edge_index, edge_attr = smile_to_graph(smiles)

        # pdb.set_trace()

        GCNData = DATA.Data(x=torch.Tensor(np.array(features)),
                                edge_index=torch.LongTensor(edge_index),
                                edge_attr = torch.LongTensor(edge_attr),
                                smiles = smiles)
                                # tCNNs_drug_matrix = tCNNs_drug_matrix)
        GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
        GCNData.target = torch.FloatTensor(np.array([cell_feature]))
        
        
    
        # model = GAT_GCN_Transformer()
        # model.load_state_dict(torch.load(model_pth), strict=True)
        # model = model.to(device)

        model = TransEDRP(config)

        model.load_state_dict(torch.load(
            infer_model, map_location='cpu'), strict=True)
        
        # device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        
        # model.to(device)

        train_batch = config["batch_size"]["train"]
        val_batch = config["batch_size"]["val"]
        test_batch = config["batch_size"]["test"]
        lr = config["lr"]
        num_epoch = config["num_epoch"]
        log_interval = config["log_interval"]

        work_dir = config["work_dir"]

        date_info = ("_infer" + dateStr()) if config["work_dir"] != "test" else ""

        # date_info = ("_" + dateStr()) if config["work_dir"] != "test" else ""
        work_dir = "./exp/" + config['marker'] + "/" + work_dir + "_"  +  date_info
        
        # if not os.path.exists("./exp/" + config['marker']):
        #     os.mkdir("./exp/" + config['marker'])
            
        # if not os.path.exists(work_dir):
        #     os.mkdir(work_dir)

        # copyfile(yaml_path, work_dir + "/")
        model_st = config["model_name"]

        
        # trainval_dataset, test_dataset = load(config)
        
        # train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset,\
        #     [
        #         int(len(trainval_dataset)*(config["dataset_type"]['train']/(config["dataset_type"]['train']+config["dataset_type"]['val']))),\
        #         len(trainval_dataset)-int(len(trainval_dataset)*(config["dataset_type"]['train']/(config["dataset_type"]['train']+config["dataset_type"]['val'])))
        #     ])
            
        # train_loader = DataLoader(train_dataset, batch_size=config['batch_size']['train'], shuffle=True, drop_last=True)
        # val_loader = DataLoader(val_dataset, batch_size=config['batch_size']['val'], shuffle=False, drop_last=False)
        
        
        
        y,_ = model.infer(GCNData)
        
        if i==0:
            print(smiles, '\t', round(y.item(),3))
            dift = y.item()-aim_ic50
        else:
            print(smiles, '\t', round(y.item()-dift,3))
            record.append([smiles, y.item()-dift])
    
    print("Mean:", np.mean([x for y,x in record]))
    print([(y, x) for y, x in record if x < aim_ic50])
    
    # import pdb;pdb.set_trace()
    
    import subprocess

    for y,x in [(y, x) for y, x in record if x < aim_ic50]:
        # 定义要执行的命令
        command = [
            'python3',  # 或 'python' 取决于你的 Python 解释器路径
            '/home/lk/project/repaint/MolPaint/temp/show_mol_from_smiles.py',
            y
        ]

        try:
            # 执行命令
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            
            # 打印输出
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            print("returncode:", result.returncode)
            
        except subprocess.CalledProcessError as e:
            # 如果命令返回非零退出状态码，捕获并打印错误
            print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
            print("stderr:", e.stderr)

class CustomDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None):
        self.data_list = data_list
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        self.process()

    @property
    def raw_file_names(self):
        return []  # 如果没有原始文件，可以返回空列表

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        if self.data_list is not None:
            self.data, self.slices = self.collate(self.data_list)
        else:
            raise RuntimeError("No data_list provided for processing.")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data.__class__()
        if self.slices is None:
            raise RuntimeError("Data has not been sliced yet.")
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(slice(slices[idx], slices[idx + 1]))
            data[key] = item[s] if item.dim() == 0 else item[s[0]]
        return data
    
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol

def get_feature(config, yaml, infer_model, x, adj):

    cell_lines_path = "data/PANCANCER_Genetic_feature.csv"
    cuda_name = 'cuda:2'
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    cell_dict, cell_features = load_cell_mut_matrix(cell_lines_path)
    record = []
    
    
    samples_int = quantize_mol(adj)
    samples_int = samples_int - 1
    samples_int[samples_int == -1] = 3   
    adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)
    x = torch.where(x > 0.5, 1, 0)
    x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)      

    import pdb;pdb.set_trace()
    
    

    # 创建一个空的 data_list
    data_list = []

    # 提取每个图的 edge_index 和 edge_attr，并生成 GCNData
    for i in tqdm(range(adj.shape[0])):
        edge_index_list = []
        edge_attr_list = []

        for j in range(adj.shape[2]):  # 遍历每个节点对
            for k in range(adj.shape[3]):
                edge_types = torch.nonzero(adj[i, :, j, k], as_tuple=False)
                if edge_types > 0:
                # for edge in edge_types:
                    edge_type = edge_types[0].item()  # 获取边的类型
                    edge_index_list.append([j, k])
                    edge_attr_list.append(edge_type)

        if edge_index_list:  # 如果 edge_index_list 非空
            edge_index = torch.tensor(edge_index_list).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list)
        else:  # 否则创建一个空的 edge_index 和 edge_attr
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)
        
        features = x[i].numpy()
        edge_index = edge_index.numpy()
        edge_attr = edge_attr.numpy()
    
        GCNData = DATA.Data(
            x=torch.Tensor(features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.LongTensor(edge_attr),
            smiles=None  # 假设此处有 smiles 变量
        )

        c_size = features.shape[0]  # 假设这是你的 c_size
        cell_feature = 1.0  # 假设这是你的 cell_feature

        GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
        GCNData.target = torch.FloatTensor(np.array([cell_feature]))

        data_list.append(GCNData)

    # 创建一个 dataset
    dataset = CustomDataset(root='/home/lk/project/repaint/MolPaint/data', data_list=data_list)
    import pdb;pdb.set_trace()
    
    
    for i in range(len(drug_list)):
        
            
        smiles = drug_list[i]
        # tCNNs_drug_matrix = canonical[i]
        
        cell_feature = cell_features[cell_dict['687799']]
        
        c_size, features, edge_index, edge_attr = smile_to_graph(smiles)

        # pdb.set_trace()

        GCNData = DATA.Data(x=torch.Tensor(np.array(features)),
                                edge_index=torch.LongTensor(edge_index),
                                edge_attr = torch.LongTensor(edge_attr),
                                smiles = smiles)
                                # tCNNs_drug_matrix = tCNNs_drug_matrix)
        GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
        GCNData.target = torch.FloatTensor(np.array([cell_feature]))
        
        
    
        # model = GAT_GCN_Transformer()
        # model.load_state_dict(torch.load(model_pth), strict=True)
        # model = model.to(device)

        model = TransEDRP(yaml)

        model.load_state_dict(torch.load(
            infer_model, map_location='cpu'), strict=True)
        
        # device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        
        # model.to(device)

        train_batch = config["batch_size"]["train"]
        val_batch = config["batch_size"]["val"]
        test_batch = config["batch_size"]["test"]
        lr = config["lr"]
        num_epoch = config["num_epoch"]
        log_interval = config["log_interval"]

        work_dir = config["work_dir"]

        date_info = ("_infer" + dateStr()) if config["work_dir"] != "test" else ""

        # date_info = ("_" + dateStr()) if config["work_dir"] != "test" else ""
        work_dir = "./exp/" + config['marker'] + "/" + work_dir + "_"  +  date_info
        
        # if not os.path.exists("./exp/" + config['marker']):
        #     os.mkdir("./exp/" + config['marker'])
            
        # if not os.path.exists(work_dir):
        #     os.mkdir(work_dir)

        # copyfile(yaml_path, work_dir + "/")
        model_st = config["model_name"]

        
        # trainval_dataset, test_dataset = load(config)
        
        # train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset,\
        #     [
        #         int(len(trainval_dataset)*(config["dataset_type"]['train']/(config["dataset_type"]['train']+config["dataset_type"]['val']))),\
        #         len(trainval_dataset)-int(len(trainval_dataset)*(config["dataset_type"]['train']/(config["dataset_type"]['train']+config["dataset_type"]['val'])))
        #     ])
            
        # train_loader = DataLoader(train_dataset, batch_size=config['batch_size']['train'], shuffle=True, drop_last=True)
        # val_loader = DataLoader(val_dataset, batch_size=config['batch_size']['val'], shuffle=False, drop_last=False)
        
        
        
        y,_ = model.infer(GCNData)
        
        record.append([smiles, y.item()])
    



def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False




# def getConfig():
#     parser = argparse.ArgumentParser(description="")
#     parser.add_argument(
#         "--config",
#         type=str,
#         required=False,
#         default="./config/Transformer_edge_concat_GDSCv2.yaml",
#         help="",
        
#     )
    
#     parser.add_argument(
#         "--model",
#         type=str,
#         required=False,
#         default="/home/lk/project/NMI_DRP/exp/TransEDRP_NCI60_m2r_30_visual_20230214110510/TransE.model",
#         help="",
#     )
    
#     args = parser.parse_args()
#     import yaml

#     with open(args.config, "r") as stream:
#         config = yaml.safe_load(stream)
#     return config, args.config, args.model




def operator(x, adj, seed, device):
    # config, yaml_path, infer_model = getConfig()
    
    config = "/home/lk/project/CLDR/CLIP_DRP/exp/TransEDRP_CLIP_KGE_then_MSE/TransEDRP_zs__20231110112037/Transedrp_CLIP_KGE_then_MSE.yaml"
    infer_model = "/home/lk/project/CLDR/CLIP_DRP/exp/TransEDRP_CLIP_KGE_then_MSE/TransEDRP_zs__20231110112037/TransE.model"
    
    with open(config, "r") as stream:
        import yaml
        yaml_path = yaml.safe_load(stream)
        
    seed_torch(seed)

    
    # main(config, yaml_path, infer_model)
    # get_feature(config, yaml_path, infer_model)
    # infer(config, yaml_path, infer_model)
    return get_feature(config, yaml_path, infer_model, x, adj)