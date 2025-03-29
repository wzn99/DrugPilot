
import sys
from rdkit import Chem
from rdkit.Chem import Draw

def visualize_smiles(smiles='Cl[CH](Br)C1#C#C#C#C#[CH]=1'):
    # 将 SMILES 转换为分子对象
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        print(f"无法解析 SMILES: {smiles}")
        return

    # 生成分子图像
    image = Draw.MolToImage(molecule, size=(300, 300))

    # 保存图像
    image_path = f'/home/lk/project/repaint/MolPaint/temp/pic/{smiles}.png'
    image.save(image_path)
    print(f"分子图像已保存到 {image_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python visualize_molecule.py <SMILES>")
    else:
        smiles = sys.argv[1]
        visualize_smiles(smiles)