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


# -------- Sampler for molecule generation tasks --------
class Sampler_mol_condition_cldr(object):
    def __init__(self, config, w=None, samples_num=1000, w=1.0):
        self.config = config
        self.device = load_device()
        self.params_x, self.params_adj = load_model_params(self.config)
        self.samples_num = 1000
        self.w = 0.0 if w is None else w
        print("self.w is ", self.w)
        
    def sample(self):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        # self.ckpt_dict_condition = load_ckpt(self.config, self.device, market='')
        self.configt = self.ckpt_dict['config']
    
        load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample-{str(self.config.controller.label['cell'])}-{str(self.config.controller.label['ic50'])}"

        
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')
        
        
        if not check_log(self.log_folder_name, self.log_name):
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)


       
        
        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device, config_train=self.configt.train)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'], self.device, config_train=self.configt.train)
        
        # self.model_x_condition = load_model_from_ckpt(self.ckpt_dict_condition['params_x'], self.ckpt_dict_condition['x_state_dict'], self.device)
        # self.model_adj_condition = load_model_from_ckpt(self.ckpt_dict_condition['params_adj'], self.ckpt_dict_condition['adj_state_dict'], self.device)

        self.sampling_fn = load_condition_sampling_fn(self.configt, self.config, self.config.sampler, self.config.sample, self.device, self.params_x, self.params_adj, self.samples_num)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
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
        
        # self.train_graph_list, _ = load_data(self.configt, get_graph_list=True)     # for init_flags
        # with open(f'{self.configt.data.dir}/{self.configt.data.data.lower()}_test_nx.pkl', 'rb') as f:
        #     self.test_graph_list = pickle.load(f)                                   # for NSPDK MMD

        # self.init_flags = init_flags(self.train_graph_list, self.configt, self.samples_num).to(self.device[0])
        self.init_flags = torch.load("./temp/temp_data/init_flags_1000.pth")
        # torch.save(self.init_flags, "./temp/temp_data/init_flags_1000.pth")
        # import pdb;pdb.set_trace()
        
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
        samples_int[samples_int == -1] = 3      # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2
        # import pdb;pdb.set_trace()
        adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)
        x = torch.where(x > 0.5, 1, 0)
        x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)      # 32, 9, 4 -> 32, 9, 5

        gen_mols, num_mols_wo_correction = gen_mol(x, adj, self.configt.data.data[0] if  type(self.configt.data.data) == list else self.configt.data.data)
        num_mols = len(gen_mols)

        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]
        
        # -------- Save generated molecules --------
        with open(os.path.join(self.log_dir, f'{self.log_name}.txt'), 'a') as f:
            f.write(f'======w:{self.w}========\n')
            for smiles in gen_smiles:
                f.write(f'{smiles}\n')
        
        # -------- Evaluation --------
        scores_1 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles_1, train=train_smiles)
        scores_2 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles_2, train=train_smiles)
        scores_3 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles_3, train=train_smiles)
        scores_4 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles_4, train=train_smiles)
        scores_5 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles_5, train=train_smiles)
        
        scores_nspdk_1 = eval_graph_list(self.test_topK_df_nx_graphs_1, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        scores_nspdk_2 = eval_graph_list(self.test_topK_df_nx_graphs_2, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        scores_nspdk_3 = eval_graph_list(self.test_topK_df_nx_graphs_3, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        scores_nspdk_4 = eval_graph_list(self.test_topK_df_nx_graphs_4, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        scores_nspdk_5 = eval_graph_list(self.test_topK_df_nx_graphs_5, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        
        logger.log(f'Number of molecules: {num_mols}')
        logger.log(f'validity w/o correction: {num_mols_wo_correction / num_mols}')
        
        for metric in ['valid', f'unique@{len(gen_smiles)}', 'FCD/Test', 'Novelty']:
            logger.log(f'{metric}: {scores_1[metric]}')
            logger.log(f'{metric}: {scores_2[metric]}')
            logger.log(f'{metric}: {scores_3[metric]}')
            logger.log(f'{metric}: {scores_4[metric]}')
            logger.log(f'{metric}: {scores_5[metric]}')
            
        logger.log(f'NSPDK MMD: {scores_nspdk_1}')
        logger.log(f'NSPDK MMD: {scores_nspdk_2}')
        logger.log(f'NSPDK MMD: {scores_nspdk_3}')
        logger.log(f'NSPDK MMD: {scores_nspdk_4}')
        logger.log(f'NSPDK MMD: {scores_nspdk_5}')
        logger.log(f'w is: {self.w}')
        logger.log('='*100)




