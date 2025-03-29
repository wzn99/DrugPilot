from rdkit import Chem
import torch
import sys
sys.path.insert(0,'/home/lk/project/repaint/MolPaint')
# gt: B([C@H](CC(C)C)NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)(O)O
# mask: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
from utils.smile_to_graph import type_check_num_atoms, construct_atomic_number_array, construct_edge_matrix
import torch
import numpy as np
import abc
from tqdm import trange
import torch.nn.functional as F
from losses import get_score_fn
from utils.graph_utils import mask_adjs, mask_x, gen_noise
from sde import VPSDE, subVPSDE
import pdb
from evaluation.stats import eval_graph_list
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx, filter_smiles_with_labels

from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol

mol = Chem.MolFromSmiles('B([C@H](CC(C)C)NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)(O)O')
type_check_num_atoms(mol, 100)

gt_x = construct_atomic_number_array(mol)
gt_adj = construct_edge_matrix(mol)


def transform(data):

    x, adj = data
    # the last place is for virtual nodes
    # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
    GDSCv2_atomic_num_list = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    x_ = np.zeros((100, 11), dtype=np.float32)

    for i in range(100):
        ind = GDSCv2_atomic_num_list.index(x[i])
        x_[i, ind] = 1.
        
    
    x = torch.tensor(x_).to(torch.float32)
    

    # single, double, triple and no-bond; the last channel is for virtual edges
    # adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
    #                         axis=0).astype(np.float32)

    x = x[:, :-1]                               # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
    # adj = torch.tensor(adj.argmax(axis=0))      # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
    # # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
    # adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
    return x, adj



device = 'cuda:0'

gt_x = torch.from_numpy(gt_x).to(device)
gt_x = F.pad(gt_x, (0, 100-gt_x.shape[0]), mode='constant', value=0)


gt_adj = torch.from_numpy(gt_adj).to(device)
gt_adj = F.pad(gt_adj, (0, 100 - gt_adj.shape[0], 0, 100 - gt_adj.shape[1]))


# .expand(100, -1, -1)

x, adj = transform([gt_x.cpu().numpy(), gt_adj.cpu().numpy()])

import pdb;pdb.set_trace()

samples_int = quantize_mol(adj)

samples_int = samples_int - 1
samples_int[samples_int == -1] = 3      # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2
# import pdb;pdb.set_trace()
        

import pdb;pdb.set_trace()

adj = torch.nn.functional.one_hot(torch.tensor(torch.from_numpy(samples_int).unsqueeze(0)), num_classes=4).permute(0, 3, 1, 2)

import pdb;pdb.set_trace()

x = torch.where(x > 0.5, 1, 0)
x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1).unsqueeze(0)      # 32, 9, 4 -> 32, 9, 5



gen_mols, num_mols_wo_correction = gen_mol(x, adj, 'GDSCv2')
num_mols = len(gen_mols)

gen_smiles = mols_to_smiles(gen_mols)
gen_smiles = [smi for smi in gen_smiles if len(smi)]

print(gen_smiles)
import pdb;pdb.set_trace()