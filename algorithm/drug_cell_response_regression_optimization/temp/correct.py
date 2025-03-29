from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import os
from tqdm import tqdm


# def correct_aromatic_rings(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return smiles

#     # 定义更通用的苯环SMARTS模式
#     benzene_pattern = Chem.MolFromSmarts('c1ccccc1')
#     Chem.Kekulize(mol)
#     # 查找并修正苯环
#     if mol.HasSubstructMatch(benzene_pattern):
#         for match in mol.GetSubstructMatches(benzene_pattern):
#             for idx in match:
#                 mol.GetAtomWithIdx(idx).SetIsAromatic(True)
#             for bond in mol.GetBonds():
#                 if bond.GetBeginAtomIdx() in match and bond.GetEndAtomIdx() in match:
#                     bond.SetBondType(Chem.BondType.AROMATIC)
#                     bond.SetIsAromatic(True)
#         # Sanitize the molecule to update its properties
#         Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
#         # return Chem.MolToSmiles(mol)
#     smiles_ = Chem.MolToSmiles(mol)
    
#     print(smiles)
#     print(smiles_)
#     return smiles

def correct_aromatic_rings(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    Chem.Kekulize(mol)
    # 遍历所有的环并检查是否为芳香环
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    print(atom_rings)
    for ring in atom_rings:
        # if is_aromatic_ring(mol, ring):

            for idx in ring:
                atom = mol.GetAtomWithIdx(idx)
                atom.SetIsAromatic(True)
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring:
                    bond.SetIsAromatic(True)
    # Sanitize the molecule to update its properties
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Chem.KekulizeException:
        print(f"KekulizeException for SMILES: {smiles}")
        return smiles
    smiles_ = Chem.MolToSmiles(mol)
    print(smiles, smiles_)
    return smiles_

# def is_aromatic_ring(mol, ring):
#     """
#     判断环是否可以被认为是芳香环。
#     当前方法简单检查是否所有原子都是 sp2 杂化的碳原子。
#     """
#     for idx in ring:
#         atom = mol.GetAtomWithIdx(idx)
#         # if atom.GetAtomicNum() != 6:  # 检查是否为碳原子
#         #     return False
#         # if atom.GetHybridization() != Chem.rdchem.HybridizationType.SP2:
#         #     return False
#     return True

# def is_aromatic_ring(mol, ring):
#     """
#     判断环是否可以被认为是芳香环。
#     当前方法简单检查是否所有原子和键都为芳香性。
#     """
#     for idx in ring:
#         atom = mol.GetAtomWithIdx(idx)
#         if not atom.GetIsAromatic():
#             return False
#     for i in range(len(ring)):
#         bond = mol.GetBondBetweenAtoms(ring[i], ring[(i+1)%len(ring)])
#         if not bond.GetIsAromatic():
#             return False
#     return True

# 读取 txt 文件中的 SMILES 字符串
txt_file = '/home/lk/project/repaint/MolPaint/temp/test.txt'

with open(txt_file, 'r') as file:
    smiles_list = [line.strip() for line in file if line.strip()]

# 修正SMILES字符串中的苯环
corrected_smiles_list = [correct_aromatic_rings(smiles) for smiles in smiles_list]

# 创建保存图片的目录
output_dir = "/home/lk/project/repaint/MolPaint/temp/pic2"
os.makedirs(output_dir, exist_ok=True)

# 每个批次的大小
batch_size = 4

# 生成分子图片
for i in tqdm(range(0, len(corrected_smiles_list), batch_size)):
    batch_smiles = corrected_smiles_list[i:i + batch_size]
    mols = [Chem.MolFromSmiles(smiles) for smiles in batch_smiles if smiles]
    mols = [mol for mol in mols if mol is not None]
    
    if not mols:
        continue

    # 使用 Draw.MolsToGridImage 生成 4x5 的图片
    img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(200, 200))

    # 保存图片
    img.save(os.path.join(output_dir, f'batch_{i//batch_size + 1}.png'))

print("图片生成完成")
