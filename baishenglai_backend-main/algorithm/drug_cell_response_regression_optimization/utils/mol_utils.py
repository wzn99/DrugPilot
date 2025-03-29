import numpy as np
import pandas as pd
import json
import networkx as nx

import re
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import pdb
from tqdm import tqdm
    
ATOM_VALENCY = {5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
bond_decoder = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC}
AN_TO_SYMBOL = {5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}


def mols_to_smiles(mols):
    return [Chem.MolToSmiles(mol) for mol in mols]


def smiles_to_mols(smiles):
    return [Chem.MolFromSmiles(s) for s in smiles]


def canonicalize_smiles(smiles):
    temp = []

    for smi in smiles:
        temp.append(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
        
    return temp
    # return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]

def load_smiles_from_single_dataset(dataset='QM9'):
    if dataset in ['QM9', 'zinc_frags_total_split']:
        col = 'SMILES1'
    elif dataset == 'ZINC250k':
        col = 'smiles'
    elif dataset in ['GDSCv2','GDSCv2_SMALL']:
        col = 'smiles'
    elif dataset in ['davis','kiba']:
        col = 'compound_iso_smiles'
    elif dataset in ['drugbank']:
        col = 'Drug1'
    else:
        raise ValueError('wrong dataset name in load_smiles')

    df = pd.read_csv(f'./algorithm/drug_cell_response_regression_optimization/data/{dataset.lower()}.csv')

    with open(f'./algorithm/drug_cell_response_regression_optimization/data/valid_idx_{dataset.lower()}.json') as f:
        test_idx = json.load(f)
    
    if dataset in ['QM9', 'GDSCv2','GDSCv2_SMALL', 'zinc_frags_total_split', 'davis','kiba', 'drugbank']:
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]
    
    train_idx = []
    
    for i in tqdm(range(len(df))):
        if i not in test_idx:
            train_idx.append(i)

    # train_idx = [i for i in range(len(df)) if i not in test_idx]
    
    return list(df[col].loc[train_idx]), list(df[col].loc[test_idx])
    
def load_smiles(dataset='QM9'):

    if type(dataset) == str:
        return load_smiles_from_single_dataset(dataset)
    elif  type(dataset) == list:
        return load_smiles_from_single_dataset(dataset[0])
        # train_list = []
        # # test_list = []
        
        # for dataset_name in dataset:
        #     train_, test_ = load_smiles_from_single_dataset(dataset_name)
        #     train_list.append(train_)
        #     # test_list.append(test_)

        # return train_list, None
        
def filter_smiles_with_labels(config, topk=3):
    
    if type(config.data.data) == list:
        config.data.data = config.data.data[0]
        
    if config.controller.base_model == "TransEDRP":
        df = pd.read_csv(f'{config.data.dir}{config.data.data.lower()}.csv')
        topK = abs(df[df['cell_name']==config.controller.label.cell]['ic50']-config.controller.label.ic50).sort_values(axis=0)[:topk]
        total_distence = sum(df.loc[topK.keys().tolist()]['ic50'].tolist())
        topK_weight = [i/total_distence for i in df.loc[topK.keys().tolist()]['ic50'].tolist()]
        topK_df = df.loc[topK.keys().tolist()]
        topK_df['weight'] = topK_weight
        return topK_df
    
    if config.controller.base_model == "TransDTA":
        df = pd.read_csv(f'{config.data.dir}{config.data.data.lower()}.csv')
        topK = abs(df[df['target_id']==config.controller.label.target]['affinity']-config.controller.label.kd).sort_values(axis=0)[:topk]
        total_distence = sum(df.loc[topK.keys().tolist()]['affinity'].tolist())
        topK_weight = [i/total_distence for i in df.loc[topK.keys().tolist()]['affinity'].tolist()]
        topK_df = df.loc[topK.keys().tolist()]
        topK_df['weight'] = topK_weight
        return topK_df
    
    if config.controller.base_model == "TransFGDRP":
        df = pd.read_csv(f'{config.data.dir}{config.data.data.lower()}.csv')
        # 找到所有符合目标片段的原始 SMILES
        matched_smiles = df[df['SMILES2'].str.contains(re.escape(config.controller.label.frag), na=False)]['SMILES1'].tolist()
        return matched_smiles

        # matched_smiles = df[df['SMILES2']==config.controller.label.frag]['SMILES1'].tolist()
        # return matched_smiles
        

        
def gen_mol(x, adj, dataset, largest_connected_comp=True):    

    x = x.detach().cpu().numpy()
    adj = adj.detach().cpu().numpy()
    
    
    
    if dataset == 'QM9':
        atomic_num_list = [6, 7, 8, 9, 0]
    elif dataset in ['GDSCv2','GDSCv2_SMALL']:
        atomic_num_list = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    elif dataset == "zinc_frags_total_split":
        atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    elif dataset == "drugbank":
        atomic_num_list = [1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 24, 26, 29, 30, 31, 33, 34, 35, 38, 43, 47, 53, 57, 64, 78, 83, 88, 0]
    else:
        atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    # mols_wo_correction = [valid_mol_can_with_seg(construct_mol(x_elem, adj_elem, atomic_num_list)) for x_elem, adj_elem in zip(x, adj)]
    # mols_wo_correction = [mol for mol in mols_wo_correction if mol is not None]
    mols, num_no_correct = [], 0
    x = x[:,:,:len(atomic_num_list)]
    

    from tqdm import tqdm
    for x_elem, adj_elem in zip(x, adj):
        
        
        mol = construct_mol(x_elem, adj_elem, atomic_num_list)
        cmol, no_correct = correct_mol(mol)
        if no_correct: num_no_correct += 1
        
        vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=largest_connected_comp)
        mols.append(vcmol)
    mols = [mol for mol in mols if mol is not None]
    return mols, num_no_correct


def construct_mol(x, adj, atomic_num_list):

    mol = Chem.RWMol()

    atoms = np.argmax(x, axis=1)
    
    if atomic_num_list is None:
        atomic_num_list = list(set(atoms))
    
    atoms_exist = (atoms != len(atomic_num_list) - 1)
    atoms = atoms[atoms_exist]          
    
      
    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    adj = np.argmax(adj, axis=0)            
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1                               
    
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
    
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


from tqdm import tqdm
import networkx as nx

def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        try:
            G = nx.Graph()
            for atom in mol.GetAtoms():
                        G.add_node(atom.GetIdx(), label=atom.GetSymbol())
                        #    atomic_num=atom.GetAtomicNum(),
                        #    formal_charge=atom.GetFormalCharge(),
                        #    chiral_tag=atom.GetChiralTag(),
                        #    hybridization=atom.GetHybridization(),
                        #    num_explicit_hs=atom.GetNumExplicitHs(),
                        #    is_aromatic=atom.GetIsAromatic())
                        
            for bond in mol.GetBonds():
                G.add_edge(bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx(),
                        label=int(bond.GetBondTypeAsDouble()))
                        #    bond_type=bond.GetBondType())
            
            nx_graphs.append(G)
        except Exception as e:
            print("Error processing molecule:", e)
            # 记录异常的分子
            # import pdb;pdb.set_trace()
                
            continue  # 继续处理下一个分子
    
    return nx_graphs


# def mols_to_nx(mols):
#     nx_graphs = []
#     for mol in tqdm(mols):
#         try:
#             G = nx.Graph()
#             for atom in mol.GetAtoms():
#                 G.add_node(atom.GetIdx(), label=atom.GetSymbol())
#                         #    atomic_num=atom.GetAtomicNum(),
#                         #    formal_charge=atom.GetFormalCharge(),
#                         #    chiral_tag=atom.GetChiralTag(),
#                         #    hybridization=atom.GetHybridization(),
#                         #    num_explicit_hs=atom.GetNumExplicitHs(),
#                         #    is_aromatic=atom.GetIsAromatic())
                        
#             for bond in mol.GetBonds():
#                 G.add_edge(bond.GetBeginAtomIdx(),
#                         bond.GetEndAtomIdx(),
#                         label=int(bond.GetBondTypeAsDouble()))
#                         #    bond_type=bond.GetBondType())
            
#             nx_graphs.append(G)
#         except Exception,e:
#             print e
#             import pdb;pdb.set_trace()
#     return nx_graphs


import re
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem

def correct_benzene_rings(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
    
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
    
        for ring in atom_rings:
            ring_atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
            bond_types = [bond.GetBondType() for bond in [mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]]
        
            # 将不合理的环替换为苯环结构
            if len(ring) == 6 and bond_types.count(Chem.rdchem.BondType.DOUBLE) > 1:
                for bond in [mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]:
                    bond.SetBondType(Chem.rdchem.BondType.AROMATIC)
                    bond.SetIsAromatic(True)
                for atom in ring_atoms:
                    atom.SetIsAromatic(True)
    
        # 对分子进行规范化和检查
        try:
            AllChem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, kekuleSmiles=False)
        except Exception as e:
            # print(smiles,'is invalid')
            # 检查环中每个原子的当前化合价
            for atom in mol.GetAtoms():
                current_valence = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
                if current_valence > 4:
                    # 如果化合价超出，将所有双键转换为单键
                    for bond in mol.GetBonds():
                        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                            bond.SetIsAromatic(False)
            try:
                AllChem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol, kekuleSmiles=False)
            except Exception as e:
                return "C"
            
    except Exception as e:
        print(f"Error!!! processing SMILES {smiles}: {e}")
        return smiles