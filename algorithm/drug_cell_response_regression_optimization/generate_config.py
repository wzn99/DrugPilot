import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_unique_smiles(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 获取不重复的smiles列值
    unique_smiles = df['smiles'].drop_duplicates().tolist()
    
    return unique_smiles

def delete_star(mol):
    # 将通配符替换为氢原子
    mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]'), replaceAll=True)[0]
    mol = Chem.RemoveHs(mol)
    return mol

def has_isolated_atoms(mol):
    # 检查分子中是否有游离的原子
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) == 0:
            return True
    return False

def filter_ic50_data(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 统计字典，key为细胞系，value为药物及其IC50值的列表
    result_dict = {}
    
    # 获取所有细胞系的名称
    cell_lines = df['cell_name'].unique()
    
    for cell_line in cell_lines:
        # 过滤出该细胞系的数据
        cell_line_data = df[df['cell_name'] == cell_line]
        
        # 计算IC50值的5%和10%分位数
        ic50_5_percentile = cell_line_data['ic50'].quantile(0.2)
        
        ic50_10_percentile = cell_line_data['ic50'].quantile(0.3)
        
        # 过滤出IC50值在5%~10%范围内的数据
        filtered_data = cell_line_data[(cell_line_data['ic50'] >= ic50_5_percentile) & 
                                       (cell_line_data['ic50'] <= ic50_10_percentile)]
        
        # 按IC50值升序排序
        sorted_data = filtered_data.sort_values(by='ic50')

        # 构建字典
        if len(sorted_data) >0:
            result_dict[str(cell_line)] = sorted_data[['smiles', 'ic50']].to_dict(orient='records')
    
    return result_dict

def read_json_as_dict(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)
    return json_dict

# 调用函数并打印结果
file_path = '/home/lk/project/repaint/MolPaint/data/gdscv2.csv'
json_file_path = '/home/lk/project/repaint/MolPaint/log/break_mols_v2.json'

unique_smiles_list = get_unique_smiles(file_path)
ic50_result_dict = filter_ic50_data(file_path)
break_mols_dict = read_json_as_dict(json_file_path)


# import pdb;pdb.set_trace()

# 将结果保存为 JSON 文件
with open('/home/lk/project/repaint/MolPaint/log/ic50_result_dict_20_30.json', 'w') as f:
    json.dump(ic50_result_dict, f, ensure_ascii=False, indent=4)


with open('/home/lk/project/repaint/MolPaint/log/break_mols_dict_20_30.json', 'w') as f:
    json.dump(break_mols_dict, f, ensure_ascii=False, indent=4)
