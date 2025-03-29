import yaml
from easydict import EasyDict as edict


def get_config(config, seed):
    # config_dir = f'./config/{config}.yaml'
    config_dir = config
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = seed

    return config