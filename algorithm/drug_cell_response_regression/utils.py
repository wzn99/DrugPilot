import os, csv
import torch
import numpy as np
from math import sqrt
from scipy import stats
# import matplotlib.pyplot as plt
import pdb
from rdkit import Chem, RDConfig
import networkx as nx
from torch_geometric import data as DATA


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

def save_cell_mut_matrix(file_path):
    f = open(file_path)
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

def preprocess_DRP(SMILES, cell_name, target_file_path=None):

    assert target_file_path != None, "Target_file_path is None!"
    
    def one_of_k_encoding(x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def atom_features(atom):
        # print(atom.GetChiralTag())
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                        one_of_k_encoding(atom.GetChiralTag(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) +
                        [atom.GetIsAromatic()]+[atom.GetAtomicNum()])

    def mol_to_graph_without_rm(mol):
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

        
        return features, edge_index, edge_attr

    cell_dict, cell_feature = save_cell_mut_matrix(target_file_path)

    data_list = []
    false_flag = []
    for smiles in SMILES:
        if smiles == "":
            print("Unable to process: ", smi)
            # return None
            false_flag.append(False)
            continue
        smi = smiles
        target = cell_feature[cell_dict[cell_name]]
        # affinity = row['affinity']
        # protein_name = row['protein_name']
        mol = Chem.MolFromSmiles(smi)
        if mol == None:
            print("Unable to process: ", smi)
            # return None
            false_flag.append(False)
            continue
        else:
            false_flag.append(True)

        x, edge_index, edge_attr = mol_to_graph_without_rm(mol)

        data = DATA.Data(
            x=torch.FloatTensor(x),
            smi=smi,
            edge_index=edge_index,
            edge_attr= torch.LongTensor(edge_attr),
            # y=torch.FloatTensor([affinity]),
            target= torch.Tensor(target.reshape(1,-1))
            # protein_name=protein_name
        )
        data_list.append(data)

        # if len(delete_list) > 0:
        # df = df.drop(delete_list, axis=0, inplace=False)
        # df.to_csv(data_path, index=False)
        # print('mol_to_graph_without_rm')
    return data_list, false_flag
