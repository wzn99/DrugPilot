import yaml
import os
from easydict import EasyDict as edict


def get_config(config, seed):
    # config_dir = f'./config/{config}.yaml'
    config_dir = config
    os.chdir('/home/data1/lk/LLM/function_call/baishenglai_backend-main')
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = seed

    return config