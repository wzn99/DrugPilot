import torch

from parsers.parser import Parser
from parsers.config import get_config

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import os
import time
import pickle
import math
import torch

from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import load_ckpt, load_data, load_seed, load_device, load_model_from_ckpt, load_model_params, \
                         load_ema_from_ckpt, load_sampling_fn, load_condition_sampling_fn, load_eval_settings
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol
from utils.plot import save_graph_list, plot_graphs_list
from evaluation.stats import eval_graph_list
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx, filter_smiles_with_labels

import sys

sys.path.insert(0,'./moses/')
from moses.metrics.metrics import get_all_metrics
from utils.mol_utils import mols_to_nx, smiles_to_mols
import copy, json
from rdkit import Chem
from tqdm import tqdm

import warnings
from rdkit import RDLogger

# 屏蔽所有警告
warnings.filterwarnings("ignore")

# 关闭RDKit日志记录器
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# -------- Sampler for molecule generation tasks --------
class Sampler_mol_condition_cldt(object):
    def __init__(self, config, w=None, samples_num=1000, device='cpu', frag_smiles=None):
        self.config = config
        self.device = load_device(device)
        self.params_x, self.params_adj = load_model_params(self.config)
        self.samples_num = samples_num
        self.w = 0.0 if w is None else w
        # print("self.w is ", self.w)
        self.frag_smiles = frag_smiles
        
    def sample(self, train_smiles = None):
        

        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        # self.ckpt_dict_condition = load_ckpt(self.config, self.device, market='')
        self.configt = self.ckpt_dict['config']
    
        load_seed(self.config.seed)


        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device, config_train=self.configt.train)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'], self.device, config_train=self.configt.train)
        
        # self.model_x_condition = load_model_from_ckpt(self.ckpt_dict_condition['params_x'], self.ckpt_dict_condition['x_state_dict'], self.device)
        # self.model_adj_condition = load_model_from_ckpt(self.ckpt_dict_condition['params_adj'], self.ckpt_dict_condition['adj_state_dict'], self.device)

        self.sampling_fn = load_condition_sampling_fn(self.configt, self.config, self.config.sampler, self.config.sample, self.device, self.params_x, self.params_adj, self.samples_num)

        # -------- Generate samples --------
        load_seed(self.config.sample.seed)

        # train_smiles, _ = load_smiles(self.configt.data.data)
        test_topK_df_1 = filter_smiles_with_labels(self.config, topk=3)
        test_topK_df_2 = filter_smiles_with_labels(self.config, topk=5)
        test_topK_df_3 = filter_smiles_with_labels(self.config, topk=10)
        test_topK_df_4 = filter_smiles_with_labels(self.config, topk=15)
        test_topK_df_5 = filter_smiles_with_labels(self.config, topk=20)
        
        
        
        test_smiles_1 = canonicalize_smiles(test_topK_df_1['smiles'].tolist())
        test_smiles_2 = canonicalize_smiles(test_topK_df_2['smiles'].tolist())
        test_smiles_3 = canonicalize_smiles(test_topK_df_3['smiles'].tolist())
        test_smiles_4 = canonicalize_smiles(test_topK_df_4['smiles'].tolist())
        test_smiles_5 = canonicalize_smiles(test_topK_df_5['smiles'].tolist())
        
        # VIP用户开源使用这个代码自己定义数量
        # self.train_graph_list, _ = load_data(self.configt, get_graph_list=True)     # for init_flags
        # with open(f'{self.configt.data.dir}/{self.configt.data.data.lower()}_test_nx.pkl', 'rb') as f:
        # with open('/home/lk/project/mol_generate/RFMG_Sampling/data/gdscv2_test_nx.pkl', 'rb') as f:
        #     self.test_graph_list = pickle.load(f)                                   # for NSPDK MMD
        # self.init_flags = init_flags(self.train_graph_list, self.configt, self.samples_num).to(f'cuda:{self.device[0]}')

        
        
        
        self.init_flags = torch.load(f"./temp/temp_data/init_flags_{self.samples_num}.pth").to(f'cuda:{self.device[0]}')
        
        # torch.save(n100, "./temp/temp_data/init_flags_100.pth")

        
        # Deal with the self.test_graph_list as test_smiles(test_topK_df)

        self.test_topK_df_nx_graphs_1 = mols_to_nx(smiles_to_mols(test_smiles_1))
        self.test_topK_df_nx_graphs_2 = mols_to_nx(smiles_to_mols(test_smiles_2))
        self.test_topK_df_nx_graphs_3 = mols_to_nx(smiles_to_mols(test_smiles_3))
        self.test_topK_df_nx_graphs_4 = mols_to_nx(smiles_to_mols(test_smiles_4))
        self.test_topK_df_nx_graphs_5 = mols_to_nx(smiles_to_mols(test_smiles_5))
        
        x, adj, _, gt_x, gt_adj, gt_keep_mask_x_, gt_keep_mask_adj_ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags, self.w)
        # x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, None)
        
        
        
        samples_int = quantize_mol(adj)

        samples_int = samples_int - 1
        samples_int[samples_int == -1] = 3      # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

        
        adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)

        x = torch.where(x > 0.5, 1, 0)
        x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)      # 32, 9, 4 -> 32, 9, 5

        gen_mols, num_mols_wo_correction = gen_mol(x, adj, self.configt.data.data[0] if  type(self.configt.data.data) == list else self.configt.data.data)
        num_mols = len(gen_mols)

        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]
        
        # after_gen
    
        
        # -------- Save generated molecules --------
        with open(os.path.join('/home/lk/project/repaint/MolPaint/log/sample_true_v2.txt'), 'a') as f:
            f.write(f'======w:{self.w}========\n')
            f.write(f'======Config:========\n')
            f.write(f'{self.config.sample.seed},{self.config.controller.config_diff_steps },{self.config.controller.label.cell},{self.config.controller.label.ic50},{self.config.controller.label.gt},{self.config.controller.label.mask},{self.frag_smiles},\n')
            from datetime import datetime
            f.write(f'======{datetime.now()}========\n')
            
            for smiles in gen_smiles:
                f.write(f'{smiles}\n')
        
        self.device[0] = f'cuda:{self.device[0]}'


        return 'ok'
        # # -------- Evaluation --------
        # scores_1 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles_1, train=train_smiles)
        # scores_2 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles_2, train=train_smiles)
        # scores_3 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles_3, train=train_smiles)
        # scores_4 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles_4, train=train_smiles)
        # scores_5 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles_5, train=train_smiles)
        
        # scores_nspdk_1 = eval_graph_list(self.test_topK_df_nx_graphs_1, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        # scores_nspdk_2 = eval_graph_list(self.test_topK_df_nx_graphs_2, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        # scores_nspdk_3 = eval_graph_list(self.test_topK_df_nx_graphs_3, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        # scores_nspdk_4 = eval_graph_list(self.test_topK_df_nx_graphs_4, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        # scores_nspdk_5 = eval_graph_list(self.test_topK_df_nx_graphs_5, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        
        # generation_results={
        #     'gen_smiles':gen_smiles,
        #     'scores_1':scores_1,
        #     'scores_2':scores_2,
        #     'scores_3':scores_3,
        #     'scores_4':scores_4,
        #     'scores_5':scores_5,
        #     'scores_nspdk_1':scores_nspdk_1,
        #     'scores_nspdk_2':scores_nspdk_2,
        #     'scores_nspdk_3':scores_nspdk_3,
        #     'scores_nspdk_4':scores_nspdk_4,
        #     'scores_nspdk_5':scores_nspdk_5,
        #     'test_smiles_1':test_smiles_1,
        #     'test_smiles_2':test_smiles_2,
        #     'test_smiles_3':test_smiles_3,
        #     'test_smiles_4':test_smiles_4,
        #     'test_smiles_5':test_smiles_5
        # }
        # return generation_results

def remove_duplicates(frag_need_masks):
    seen = set()
    unique_frags = []
    
    for frag in frag_need_masks:
        frag_smiles = frag['frag_smiles']
        sorted_frag_id = tuple(sorted(frag['frag_id']))  # 将 frag_id 排序并转换为元组
        identifier = (frag_smiles, sorted_frag_id)
        
        if identifier not in seen:
            seen.add(identifier)
            unique_frags.append(frag)
    
    return unique_frags

def drug_cell_response_regression_generation(model_type = 'molgen', condition_strength=1.0, seed=42, gen_number=1000, timestep = 1000, ic50 = 0.4, cell_line = '1290730', device='cpu'):
    
    # from models.CLDR import get_feature, operator
    
    # asd = operator( torch.randint(0,10,(100,100,10))/10,  torch.randint(0,5,(100,100,100)), seed, device)
    

    
    seed = 42
    config = get_config('./config/sample_gdscv2.yaml', seed)
        
    if model_type == 'molgen':

        # config
        config.sample.seed = seed if seed is not None else config.sample.seed
        config.controller.config_diff_steps = timestep if timestep is not None else config.controller.config_diff_steps
        config.controller.label.cell = cell_line if cell_line is not None else config.controller.label.cell
        config.controller.label.ic50 = ic50 if ic50 is not None else config.controller.label.ic50


        sampler = Sampler_mol_condition_cldr(config, w=condition_strength, samples_num=gen_number,  device=device)
        return sampler.sample()
    elif model_type == 'multimolgen':
        
        # 加载保存的 JSON 文件
        with open('/home/lk/project/repaint/MolPaint/log/ic50_result_dict.json', 'r', encoding='utf-8') as f:
            ic50_result_dict = json.load(f)

        with open('/home/lk/project/repaint/MolPaint/log/break_mols_dict.json', 'r', encoding='utf-8') as f:
            break_mols_dict = json.load(f)
        
        train_smiles, _ = load_smiles(config.data.data)
        train_smiles = canonicalize_smiles(train_smiles)
        
        # 遍历每种细胞系及其对应的药物
        for cell_line, drugs in tqdm(ic50_result_dict.items()):
            for drug in drugs:
                smiles = drug['smiles']
                if break_mols_dict.get(smiles) is None:
                    continue
                # print(smiles)
                original_ic50 = drug['ic50']
                reduced_ic50 = original_ic50 * 0.8
                
                # 生成配置对象的副本
                current_config = copy.deepcopy(config)
                
                # 修改配置
                current_config.sample.seed = seed if seed is not None else current_config.sample.seed
                current_config.controller.config_diff_steps = timestep if timestep is not None else current_config.controller.config_diff_steps
                current_config.controller.label.cell = int(cell_line)
                current_config.controller.label.ic50 = round(reduced_ic50,2)
                current_config.controller.label.gt = smiles
                
                
                
                mol = Chem.MolFromSmiles(smiles)
                atoms_len = len(mol.GetAtoms())
                
                frag_need_masks = break_mols_dict[smiles][1:][0]
                # 去重
                frag_need_masks = remove_duplicates(frag_need_masks)

                # import pdb;pdb.set_trace()
                for frag_need_mask in frag_need_masks:
                    frag_smiles = frag_need_mask['frag_smiles']
                    frag_idx = frag_need_mask['frag_id']
                    mask =  [0] * atoms_len
                    for idx in frag_idx:
                        mask[idx] = 1
                    current_config.controller.label.mask = mask
                    # print(mask)
                    sampler = Sampler_mol_condition_cldr(current_config, w=condition_strength, samples_num=gen_number,  device=device)
                    sampler.sample(train_smiles=train_smiles)
    else:
        return None
    
if __name__ == '__main__':
    

    import torch

    # 检查系统中可用的CUDA设备数量
    num_devices = torch.cuda.device_count()
    print("可用的CUDA设备数量：", num_devices)

    # 打印每个设备的信息
    for i in range(num_devices):
        print("CUDA 设备 {}: {}".format(i, torch.cuda.get_device_name(i)))

    # model_type = 'molgen'
    model_type = 'multimolgen'
    device = 'cuda:1'
    
    # gen_number
    # 100,200,300,400,500,600,700,800,900,1000, 1500,2000,5000,10000
    # 显存需要至少1G/100 +470
    generation_results = drug_cell_response_regression_generation(model_type=model_type, condition_strength=5.0, seed=42, gen_number=100, timestep = 300, ic50 = None, cell_line = None, device=device)
    
    