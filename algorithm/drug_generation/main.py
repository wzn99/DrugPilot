import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.append(f'{current_dir}/moses/')

from parsers.parser import Parser
from parsers.config import get_config

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import os
os.chdir('/home/data1/lk/LLM/function_call/baishenglai_backend-main')
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
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx, \
    filter_smiles_with_labels

import sys

from moses.metrics.metrics import get_all_metrics
from utils.mol_utils import mols_to_nx, smiles_to_mols


# -------- Sampler for molecule generation tasks --------
class Sampler_mol_condition_cldr(object):
    def __init__(self, config, w=None, samples_num=1000, device='cpu'):
        self.config = config
        self.device = load_device(device)
        self.params_x, self.params_adj = load_model_params(self.config)
        self.samples_num = samples_num
        self.w = 0.0 if w is None else w
        print("self.w is ", self.w)


    def sample(self):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        # self.ckpt_dict_condition = load_ckpt(self.config, self.device, market='')
        self.configt = self.ckpt_dict['config']

        load_seed(self.config.seed)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device,
                                            config_train=self.configt.train)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'],
                                              self.device, config_train=self.configt.train)

        # self.model_x_condition = load_model_from_ckpt(self.ckpt_dict_condition['params_x'], self.ckpt_dict_condition['x_state_dict'], self.device)
        # self.model_adj_condition = load_model_from_ckpt(self.ckpt_dict_condition['params_adj'], self.ckpt_dict_condition['adj_state_dict'], self.device)

        self.sampling_fn = load_condition_sampling_fn(self.configt, self.config, self.config.sampler,
                                                      self.config.sample, self.device, self.params_x, self.params_adj,
                                                      self.samples_num)

        # -------- Generate samples --------
        load_seed(self.config.sample.seed)

        train_smiles, _ = load_smiles(self.configt.data.data)
        test_topK_df_1 = filter_smiles_with_labels(self.config, topk=3)
        test_topK_df_2 = filter_smiles_with_labels(self.config, topk=5)
        test_topK_df_3 = filter_smiles_with_labels(self.config, topk=10)
        test_topK_df_4 = filter_smiles_with_labels(self.config, topk=15)
        test_topK_df_5 = filter_smiles_with_labels(self.config, topk=20)

        train_smiles = canonicalize_smiles(train_smiles)

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

        self.init_flags = torch.load(f"./algorithm/drug_generation/temp/temp_data/init_flags_{self.samples_num}.pth").to(f'cuda:{self.device[0]}')

        # torch.save(n100, "./temp/temp_data/init_flags_100.pth")

        # Deal with the self.test_graph_list as test_smiles(test_topK_df)

        self.test_topK_df_nx_graphs_1 = mols_to_nx(smiles_to_mols(test_smiles_1))
        self.test_topK_df_nx_graphs_2 = mols_to_nx(smiles_to_mols(test_smiles_2))
        self.test_topK_df_nx_graphs_3 = mols_to_nx(smiles_to_mols(test_smiles_3))
        self.test_topK_df_nx_graphs_4 = mols_to_nx(smiles_to_mols(test_smiles_4))
        self.test_topK_df_nx_graphs_5 = mols_to_nx(smiles_to_mols(test_smiles_5))

        x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags, self.w)
        # x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, None)

        samples_int = quantize_mol(adj)

        samples_int = samples_int - 1
        samples_int[samples_int == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

        adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)
        x = torch.where(x > 0.5, 1, 0)
        x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)  # 32, 9, 4 -> 32, 9, 5
        import time
        # time.sleep(20)
        gen_mols, num_mols_wo_correction = gen_mol(x, adj, self.configt.data.data[0] if type(
            self.configt.data.data) == list else self.configt.data.data)
        num_mols = len(gen_mols)
        print('generation completed')
        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]
        # # -------- Save generated molecules --------
        # with open(os.path.join(self.log_dir, f'{self.log_name}.txt'), 'a') as f:
        #     f.write(f'======w:{self.w}========\n')
        #     for smiles in gen_smiles:
        #         f.write(f'{smiles}\n')

        self.device[0] = f'cuda:{self.device[0]}'

        # -------- Evaluation --------
        print('start evaluation')
        n_jobs = 1
        scores_1 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=n_jobs,
                                   test=test_smiles_1, train=train_smiles)
        scores_2 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=n_jobs,
                                   test=test_smiles_2, train=train_smiles)
        scores_3 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=n_jobs,
                                   test=test_smiles_3, train=train_smiles)
        scores_4 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=n_jobs,
                                   test=test_smiles_4, train=train_smiles)
        scores_5 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=n_jobs,
                                   test=test_smiles_5, train=train_smiles)
        print('all metric got')
        scores_nspdk_1 = eval_graph_list(self.test_topK_df_nx_graphs_1, mols_to_nx(gen_mols), methods=['nspdk'])[
            'nspdk']
        scores_nspdk_2 = eval_graph_list(self.test_topK_df_nx_graphs_2, mols_to_nx(gen_mols), methods=['nspdk'])[
            'nspdk']
        scores_nspdk_3 = eval_graph_list(self.test_topK_df_nx_graphs_3, mols_to_nx(gen_mols), methods=['nspdk'])[
            'nspdk']
        scores_nspdk_4 = eval_graph_list(self.test_topK_df_nx_graphs_4, mols_to_nx(gen_mols), methods=['nspdk'])[
            'nspdk']
        scores_nspdk_5 = eval_graph_list(self.test_topK_df_nx_graphs_5, mols_to_nx(gen_mols), methods=['nspdk'])[
            'nspdk']
        print('eval completed')
        generation_results = {
            'gen_smiles': gen_smiles,
            'scores_1': scores_1['FCD/Test'],
            'scores_2': scores_2['FCD/Test'],
            'scores_3': scores_3['FCD/Test'],
            'scores_4': scores_4['FCD/Test'],
            'scores_5': scores_5['FCD/Test'],
            'scores_nspdk_1': scores_nspdk_1,
            'scores_nspdk_2': scores_nspdk_2,
            'scores_nspdk_3': scores_nspdk_3,
            'scores_nspdk_4': scores_nspdk_4,
            'scores_nspdk_5': scores_nspdk_5,
            'test_smiles_1': test_smiles_1,
            'test_smiles_2': test_smiles_2,
            'test_smiles_3': test_smiles_3,
            'test_smiles_4': test_smiles_4,
            'test_smiles_5': test_smiles_5
        }
        print('evaluation completed')

        return generation_results


def drug_cell_response_regression_generation(cell_line, zscore):
    """
    Generates drug molecules whose interaction with the specified cell line results in the given z-score.

    Args:
        cell_line (str): The name of the cell line.
        zscore (float): The z-score representing the interaction between the generated drug molecules and the cell line.

    """

    # 假数据
    # return ['C(OC(C(F)(F)F)C(F)(F)F)F', 'Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1']

    model_type='molgen'
    condition_strength=1.0
    seed=42
    gen_number=100
    timestep=100
    device='cuda:0'
    
    cell_line = int(cell_line)
    zscore = float(zscore)
    try:
        # print(locals())
        ic50 = 1 / (1 + pow(math.exp(float(zscore)), -0.1))
        config = get_config('./algorithm/drug_generation/config/sample_gdscv2.yaml', 42)
        # config
        config.sample.seed = seed
        config.controller.config_diff_steps = timestep
        config.controller.label.cell = cell_line
        config.controller.label.ic50 = ic50

        sampler = Sampler_mol_condition_cldr(config, w=condition_strength, samples_num=gen_number, device=device)

        gen_smiles = sampler.sample()['gen_smiles']
        output = {}
        output['gen_smiles'] = gen_smiles[:10]
        return output
    except IndexError as e:
        return 'invalid cell name'
    except ZeroDivisionError and OverflowError as e:
        return 'invalid z-score'


if __name__ == '__main__':
    import torch

    # 检查系统中可用的CUDA设备数量
    num_devices = torch.cuda.device_count()
    print("可用的CUDA设备数量：", num_devices)

    # 打印每个设备的信息
    for i in range(num_devices):
        print("CUDA 设备 {}: {}".format(i, torch.cuda.get_device_name(i)))

    model_type = 'molgen'
    device = 'cuda:0'
    cell_line='1290730'
    ic50=0.4
    # gen_number
    # 100,200,300,400,500,600,700,800,900,1000, 1500,2000,5000,10000
    # 显存需要至少1G/100 +470
    generation_results = drug_cell_response_regression_generation(cell_line='684059', zscore='-4.054651')
    print(generation_results)
