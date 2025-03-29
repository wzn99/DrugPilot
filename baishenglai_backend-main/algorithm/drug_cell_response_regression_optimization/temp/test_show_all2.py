import sys
sys.path.insert(0,'/home/lk/project/repaint/MolPaint')
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import matplotlib.pyplot as plt
import numpy as np
    
from datetime import datetime, timedelta, timezone
from moses.metrics.metrics import get_all_metrics
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx, filter_smiles_with_labels
from utils.mol_utils import mols_to_nx, smiles_to_mols
from evaluation.stats import eval_graph_list
import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole


from tqdm import tqdm
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType
import matplotlib.pyplot as plt

def mol_to_nx(mol):
    """Convert an RDKit molecule to a NetworkX graph."""
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondTypeAsDouble())
    return G

def graph_edit_distance(G1, G2):
    """Calculate the graph edit distance between two graphs."""
    return nx.graph_edit_distance(G1, G2)

with open("/home/lk/project/repaint/MolPaint/temp/test.txt", "r") as f:
    smiles_list = f.read().strip().split('\n')



# 目标分子和指纹
target_smiles = smiles_list[-1]
target_mol = Chem.MolFromSmiles(target_smiles)
target_graph = mol_to_nx(target_mol)

# import pdb;pdb.set_trace()
smiles_list = [x for x in set(smiles_list[:-1])]
print('len of generated smiles:',len(smiles_list))

similarities = []

def calculate_and_plot_similarities(smiles_list, target_smiles, output_dir):
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_graph = mol_to_nx(target_mol)
    similarities = []

    for idx, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list)):
        fig, axs = plt.subplots(1, 2, figsize=(10, 2))

        # import pdb;pdb.set_trace()
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol)
                mol_graph = mol_to_nx(mol)
                distance = graph_edit_distance(target_graph, mol_graph)
                similarity = 1 / (1 + distance)  # 将距离转换为相似性
                similarities.append((smiles, similarity))

                axs[0].imshow(img)
                axs[0].axis('off')
                axs[0].set_title(f'Molecule {idx + 1}\nSimilarity: {similarity:.2f}')

                axs[1].imshow(Draw.MolToImage(target_mol))
                axs[1].axis('off')
                axs[1].set_title('Target Molecule')

                plt.tight_layout()
                plt.savefig(f'{output_dir}/show_part_{idx + 1}.png')
                plt.close(fig)

    # 获取当前北京时间
    beijing_time = datetime.now(timezone(timedelta(hours=8)))

    # 格式化时间戳
    timestamp = beijing_time.strftime("%Y%m%d_%H%M%S")


    # 将相似性结果保存到CSV文件
    df = pd.DataFrame(similarities, columns=['SMILES', 'Similarity'])
    df.to_csv(f'/home/lk/project/repaint/MolPaint/temp/similarities_{timestamp}.csv', index=False)


calculate_and_plot_similarities(smiles_list, target_smiles, '/home/lk/project/repaint/MolPaint/temp')
