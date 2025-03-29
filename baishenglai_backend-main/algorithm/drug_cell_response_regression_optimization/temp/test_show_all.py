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

with open("/home/lk/project/repaint/MolPaint/temp/test.txt", "r") as f:
    smiles_list = f.read().strip().split('\n')

target_smiles = smiles_list[0]
target_mol = Chem.MolFromSmiles(target_smiles)
target_fp = FingerprintMols.FingerprintMol(target_mol)


# import pdb;pdb.set_trace()
smiles_list = [x for x in set(smiles_list[1:])]
print('len of generated smiles:',len(smiles_list))

if len(smiles_list)>=1000:
    # 分成10份
    num_parts = 10
    smiles_chunks = np.array_split(smiles_list, num_parts)

    # 计算相似度并保存分子图像
    similarities = []

    for i, chunk in enumerate(smiles_chunks):
        fig, axs = plt.subplots(len(chunk), 2, figsize=(10, len(chunk) * 2))

        for idx, smiles in enumerate(chunk):
            if smiles != "":
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    img = Draw.MolToImage(mol)
                    mol_fp = FingerprintMols.FingerprintMol(mol)
                    similarity = DataStructs.FingerprintSimilarity(target_fp, mol_fp)
                    similarities.append((smiles, similarity))

                    axs[idx, 0].imshow(img)
                    axs[idx, 0].axis('off')
                    axs[idx, 0].set_title(f'Molecule {idx + 1}')

                    axs[idx, 1].imshow(Draw.MolToImage(target_mol))
                    axs[idx, 1].axis('off')
                    axs[idx, 1].set_title('Target Molecule')

        plt.tight_layout()
        plt.savefig(f'/home/lk/project/repaint/MolPaint/temp/show_part_{i + 1}.png')
        plt.close(fig)


    # 获取当前北京时间
    beijing_time = datetime.now(timezone(timedelta(hours=8)))

    # 格式化时间戳
    timestamp = beijing_time.strftime("%Y%m%d_%H%M%S")

    # 保存相似度为带时间戳的CSV文件
    similarity_df = pd.DataFrame(similarities, columns=['SMILES', 'Similarity'])
    similarity_df.to_csv(f'/home/lk/project/repaint/MolPaint/temp/csv/similarities_{timestamp}.csv', index=False)

    test_smiles_1 = canonicalize_smiles([target_smiles])
    # import pdb;pdb.set_trace()

    # scores_1 = get_all_metrics(gen=smiles_list, k=len(smiles_list), device='cuda:0', n_jobs=1, test=test_smiles_1, train=smiles_list)
    # print(scores_1)

    # import pdb;pdb.set_trace()

    test_topK_df_nx_graphs_1 = mols_to_nx(smiles_to_mols(test_smiles_1))
    scores_nspdk_1 = eval_graph_list(test_topK_df_nx_graphs_1, mols_to_nx(smiles_to_mols(smiles_list)), methods=['nspdk'])['nspdk']
    print(scores_nspdk_1)

    # scores_nspdk_1 = eval_graph_list(test_topK_df_nx_graphs_1, test_topK_df_nx_graphs_1, methods=['nspdk'])['nspdk']
    # print(scores_nspdk_1)
else:
        # 分成10份
    num_parts = 1
    smiles_chunks = np.array_split(smiles_list, num_parts)

    # 计算相似度并保存分子图像
    similarities = []

    for i, chunk in enumerate(smiles_chunks):
        fig, axs = plt.subplots(len(chunk), 2, figsize=(10, len(chunk) * 2))

        for idx, smiles in enumerate(chunk):
            if smiles != "":
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    img = Draw.MolToImage(mol)
                    mol_fp = FingerprintMols.FingerprintMol(mol)
                    similarity = DataStructs.FingerprintSimilarity(target_fp, mol_fp)
                    similarities.append((smiles, similarity))

                    axs[idx, 0].imshow(img)
                    axs[idx, 0].axis('off')
                    axs[idx, 0].set_title(f'Molecule {idx + 1} {smiles}')

                    axs[idx, 1].imshow(Draw.MolToImage(target_mol))
                    axs[idx, 1].axis('off')
                    axs[idx, 1].set_title('Target Molecule')

        plt.tight_layout()
        plt.savefig(f'/home/lk/project/repaint/MolPaint/temp/show.png')
        plt.close(fig)


    # 获取当前北京时间
    beijing_time = datetime.now(timezone(timedelta(hours=8)))

    # 格式化时间戳
    timestamp = beijing_time.strftime("%Y%m%d_%H%M%S")

    # 保存相似度为带时间戳的CSV文件
    similarity_df = pd.DataFrame(similarities, columns=['SMILES', 'Similarity'])
    similarity_df.to_csv(f'/home/lk/project/repaint/MolPaint/temp/similarities_{timestamp}.csv', index=False)

    test_smiles_1 = canonicalize_smiles([target_smiles])
    # import pdb;pdb.set_trace()

    # scores_1 = get_all_metrics(gen=smiles_list, k=len(smiles_list), device='cuda:0', n_jobs=1, test=test_smiles_1, train=smiles_list)
    # print(scores_1)

    # import pdb;pdb.set_trace()

    test_topK_df_nx_graphs_1 = mols_to_nx(smiles_to_mols(test_smiles_1))
    scores_nspdk_1 = eval_graph_list(test_topK_df_nx_graphs_1, mols_to_nx(smiles_to_mols(smiles_list)), methods=['nspdk'])['nspdk']
    print(scores_nspdk_1)
