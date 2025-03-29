from rdkit import Chem
from rdkit.Chem import Draw

def draw_molecule(smiles, output_file):
    # 解析 SMILES 表示的分子
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # 绘制原始分子图并保存为 PNG 文件
    drawer = Draw.MolDraw2DCairo(500, 500)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    drawer.WriteDrawingText(output_file)

def draw_molecule_with_highlights(smiles, output_file, mask):
    # 解析 SMILES 表示的分子
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # 标注每个原子
    for atom in mol.GetAtoms():
        atom.SetProp('atomLabel', str(atom.GetIdx()))

    print('mol origin atom list:', mask)
    
    # 高亮标记的原子
    highlight_atoms = [i for i, x in enumerate(mask) if x == 1]
    print('Highlighted atoms:', highlight_atoms)
    
    # 绘制分子图并保存为 PNG 文件
    drawer = Draw.MolDraw2DCairo(500, 500)
    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
    drawer.FinishDrawing()
    drawer.WriteDrawingText(output_file)

# 示例 SMILES 字符串
smiles = "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"

# 创建掩码列表
mol = Chem.MolFromSmiles(smiles)
mask = [0] * len(mol.GetAtoms())
mask = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
# mask[16:24] = [1] * (24-16)

print("mask:", mask)

# 生成原始分子图像
original_output_file = "/home/lk/project/repaint/MolPaint/temp/molecule_original.png"
draw_molecule(smiles, original_output_file)
print(f"Original molecule image saved to {original_output_file}")

# 生成高亮标记原子的分子图像
highlighted_output_file = "/home/lk/project/repaint/MolPaint/temp/molecule_highlighted.png"
draw_molecule_with_highlights(smiles, highlighted_output_file, mask)
print(f"Highlighted molecule image saved to {highlighted_output_file}")
