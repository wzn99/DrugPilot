import torch
import random
import numpy as np

from algorithm.drug_cell_response_regression_optimization.models.ScoreNetwork_A import ScoreNetworkA
from algorithm.drug_cell_response_regression_optimization.models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH

from algorithm.drug_cell_response_regression_optimization.models.ScoreNetwork_A_control import ScoreNetworkA_control
from algorithm.drug_cell_response_regression_optimization.models.ScoreNetwork_X_control import ScoreNetworkX_control

from algorithm.drug_cell_response_regression_optimization.sde import VPSDE, VESDE, subVPSDE

from algorithm.drug_cell_response_regression_optimization.controller import TransEDRP, TransFGDRP, TransDTA

from algorithm.drug_cell_response_regression_optimization.losses import get_sde_loss_fn
from algorithm.drug_cell_response_regression_optimization.solver import get_pc_sampler, S4_solver, get_pc_conditional_sampler
from algorithm.drug_cell_response_regression_optimization.evaluation.mmd import gaussian, gaussian_emd
from algorithm.drug_cell_response_regression_optimization.utils.ema import ExponentialMovingAverage
import pdb

def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def load_device(device=None):
    if torch.cuda.is_available():
        if isinstance(device, str):
            device = [str(device.split(':')[-1])]
        else:
            device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
        
    
    return device


def add_control_to_nets(params, config_train):
    params_ = params.copy()
    model_type = params_.pop('model_type', None)
    

    
    if model_type == 'ScoreNetworkX':
        model = ScoreNetworkX_control(**params_)
    elif model_type == 'ScoreNetworkA':
        model = ScoreNetworkA_control(**params_)
    return model


def load_model(params, config_train=None):
    

        
    params_ = params.copy()
    model_type = params_.pop('model_type', None)
    

    if config_train and config_train.get('task'):
        return add_control_to_nets(params, config_train)
    
    if model_type == 'ScoreNetworkX':
        model = ScoreNetworkX(**params_)
    elif model_type == 'ScoreNetworkX_GMH':
        model = ScoreNetworkX_GMH(**params_)
    elif model_type == 'ScoreNetworkA':
        model = ScoreNetworkA(**params_)
    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")
    return model


def load_model_optimizer(params, config_train, device):
    model = load_model(params, config_train)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
            # if isinstance(model,torch.nn.DataParallel):
            #     model = model.module
            #     model = model.cuda(device=device[0])
        elif len(device) == 1:
            model = model.to(f'cuda:{device[0]}')
    #  torch.optim.Adam(filter(lambda p: p.requires_grad,cnn.parameters()), lr=learning_rate)
    optimizer = torch.optim.Adam( model.parameters(), lr=config_train.lr, 
                                    weight_decay=config_train.weight_decay)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    
    return model, optimizer, scheduler


def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema


def load_data(config, get_graph_list=False):
    if type(config.data.data) == list or (type(config.data.data) == str and config.data.data in ['QM9', 'ZINC250k', 'GDSCv2', 'GDSCv2_SMALL', 'davis', 'kiba', 'BindingDB_Kd', 'zinc_frags_total_split']):
        from utils.data_loader_mol import dataloader
        return dataloader(config, get_graph_list)
    else:
        from utils.data_loader import dataloader
        return dataloader(config, get_graph_list)


def load_batch(batch, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    if len(batch) == 2:
        return x_b, adj_b
    else:
        label_b = batch[2].to(device_id)
        return x_b, adj_b, label_b


def load_sde(config_sde):
    sde_type = config_sde.type
    beta_min = config_sde.beta_min
    beta_max = config_sde.beta_max
    num_scales = config_sde.num_scales

    if sde_type == 'VP':
        sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    elif sde_type == 'VE':
        sde = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_scales)
    elif sde_type == 'subVP':
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    else:
        raise NotImplementedError(f"SDE class {sde_type} not yet supported.")
    return sde

def load_condition_predictor(config_train, config_controller, device_id, params_x, params_adj):
    # with open(config_controller.model, "r") as stream:
    #     import yaml
    #     model_config = yaml.safe_load(stream)
    # import pdb;pdb.set_trace()
    
    if config_controller.base_model == 'TransEDRP':
        controller = TransEDRP(config_controller, params_x, params_adj, device_id, None)
        # add pth to modelnn
        controller.load_state_dict(torch.load(config_controller.pth_dir, map_location=torch.device('cpu'))['model_state_dict'], strict=True)
        controller.get_cell_matrix(config_controller.cell_csv_path)
        return controller
    
    elif config_controller.base_model == 'TransDTA':
        controller = TransDTA(config_controller, params_x, params_adj, device_id, config_train, config_controller.target_txt_path, PRECISION=config_train.PRECISION)
        # add pth to modelnn
        controller.load_state_dict(torch.load(config_controller.pth_dir, map_location=torch.device('cpu'))['model_state_dict'], strict=True)
        controller.get_target_feature(config_controller.target_txt_path)
        return controller
    
    elif config_controller.base_model== 'TransDDI':
        controller = TransDDI(self.config.controller, self.params_x, self.params_adj, f'cuda:{self.device[0]}', self.config.train)
        controller.load_state_dict(torch.load(config_controller.pth_dir, map_location=torch.device('cpu'))['model_state_dict'], strict=True)
        return controller
    
    elif config_controller.base_model == 'TransFGDRP':
        controller = TransFGDRP(config_controller, params_x, params_adj, device_id, None)
        # add pth to modelnn
        controller.load_state_dict(torch.load(config_controller.pth_dir, map_location=torch.device('cpu'))['model_state_dict'], strict=True)
        return controller
    
            
        # self.dta_label_embedding = TransDTA(self.config.controller, self.params_x, self.params_adj, f'cuda:{self.device[0]}', self.config.train, self.target_list, PRECISION=self.config.PRECISION)
        # self.dta_label_embedding.load_state_dict(torch.load(self.config.controller.cldr_ckpt)['model_state_dict'], strict = True)
        


def load_loss_fn(config):
    reduce_mean = config.train.reduce_mean
    

    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)
    
    loss_fn = get_sde_loss_fn(sde_x, sde_adj, train=True, reduce_mean=reduce_mean, continuous=True, 
                                likelihood_weighting=False, eps=config.train.eps)
    return loss_fn


def load_sampling_fn(config_train, config_module, config_sample, device):
    sde_x = load_sde(config_train.sde.x)
    sde_adj = load_sde(config_train.sde.adj)
    max_node_num  = config_train.data.max_node_num

    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device

    if config_module.predictor == 'S4':
        get_sampler = S4_solver
    else:
        get_sampler = get_pc_sampler

    if config_train.data.data in ['QM9', 'ZINC250k', 'GDSCv2', 'GDSCv2_SMALL', 'zinc_frags_total_split', 'davis', 'drugbank']:
        shape_x = (1000, max_node_num, config_train.data.max_feat_num)
        shape_adj = (1000, max_node_num, max_node_num)
    else:
        shape_x = (config_train.data.batch_size, max_node_num, config_train.data.max_feat_num)
        shape_adj = (config_train.data.batch_size, max_node_num, max_node_num)
        
    sampling_fn = get_sampler(sde_x=sde_x, sde_adj=sde_adj, shape_x=shape_x, shape_adj=shape_adj, 
                                predictor=config_module.predictor, corrector=config_module.corrector,
                                snr=config_module.snr, scale_eps=config_module.scale_eps, 
                                n_steps=config_module.n_steps, 
                                probability_flow=config_sample.probability_flow, 
                                continuous=True, denoise=config_sample.noise_removal, 
                                eps=config_sample.eps, device=device_id)
    return sampling_fn


def load_condition_sampling_fn(config_train, config, config_module, config_sample, device, params_x, params_adj, samples_num):
    sde_x = load_sde(config_train.sde.x)
    sde_adj = load_sde(config_train.sde.adj)
    max_node_num  = config_train.data.max_node_num
    
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    controller = load_condition_predictor(config_train, config.controller, device_id, params_x, params_adj)
    controller.to(device_id)

    if config_train.data.data in ['GDSCv2', ['GDSCv2'], 'GDSCv2_SMALL', ['GDSCv2', 'QM9'],'zinc_frags_total_split','davis', ['davis', 'QM9'], 'kiba', ['kiba', 'QM9'], 'drugbank', ['drugbank', 'QM9']]:
        shape_x = (samples_num, max_node_num, config_train.data.max_feat_num)
        shape_adj = (samples_num, max_node_num, max_node_num)
        get_sampler = get_pc_conditional_sampler
        
    
    sampling_fn = get_sampler(controller=controller, sde_x=sde_x, sde_adj=sde_adj, shape_x=shape_x, shape_adj=shape_adj, 
                                predictor=config_module.predictor, corrector=config_module.corrector,
                                snr=config_module.snr, scale_eps=config_module.scale_eps, 
                                n_steps=config_module.n_steps, 
                                probability_flow=config_sample.probability_flow, 
                                continuous=True, denoise=config_sample.noise_removal, 
                                eps=config_sample.eps, config=config, device=device_id)
    return sampling_fn


def load_model_params(config):
  
    config_m = config.model
    max_feat_num = config.data.max_feat_num

    if 'GMH' in config_m.x:
        params_x = {'model_type': config_m.x, 'max_feat_num': max_feat_num, 'depth': config_m.depth, 
                    'nhid': config_m.nhid, 'num_linears': config_m.num_linears,
                    'c_init': config_m.c_init, 'c_hid': config_m.c_hid, 'c_final': config_m.c_final, 
                    'adim': config_m.adim, 'num_heads': config_m.num_heads, 'conv':config_m.conv}
    else:
        params_x = {'model_type':config_m.x, 'max_feat_num':max_feat_num, 'depth':config_m.depth, 'nhid':config_m.nhid}
    params_adj = {'model_type':config_m.adj, 'max_feat_num':max_feat_num, 'max_node_num':config.data.max_node_num, 
                    'nhid':config_m.nhid, 'num_layers':config_m.num_layers, 'num_linears':config_m.num_linears, 
                    'c_init':config_m.c_init, 'c_hid':config_m.c_hid, 'c_final':config_m.c_final, 
                    'adim':config_m.adim, 'num_heads':config_m.num_heads, 'conv':config_m.conv}
    return params_x, params_adj


def load_ckpt(config, device, ts=None, return_ckpt=False, market=''):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    ckpt_dict = {}
    if ts is not None:
        config.ckpt = ts
        
    # path = "/home/lk/project/mol_generate/GDSS/checkpoints/['davis', 'QM9']/Apr10-13:17:40_174.pth"
    path = f'./algorithm/drug_cell_response_regression_optimization/checkpoints/{config.data.data}/{config.ckpt}{market}.pth'
    # path = f'/home/nas/lk/GDSS/checkpoints/{config.data.data}/{config.ckpt}{market}.pth'

    if device[0] == 'cpu':
        ckpt = torch.load(path, map_location=f'cpu')
    else:
        ckpt = torch.load(path, map_location=device_id)
        
    
    # print(f'{path} loaded')
    ckpt_dict= {'config': ckpt['model_config'], 'params_x': ckpt['params_x'], 'x_state_dict': ckpt[f'x_state_dict{market}'],
                'params_adj': ckpt['params_adj'], 'adj_state_dict': ckpt[f'adj_state_dict{market}']}
    if config.sample.use_ema:
        ckpt_dict['ema_x'] = ckpt[f'ema_x{market}']
        ckpt_dict['ema_adj'] = ckpt[f'ema_adj{market}']
    if return_ckpt:
        ckpt_dict['ckpt'] = ckpt
    return ckpt_dict


def load_model_from_ckpt(params, state_dict, device, config_train=None):
    model = load_model(params, config_train=config_train)
    if 'module.' in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    return model


def load_eval_settings(data, orbit_on=True):
    # Settings for generic graph generation
    methods = ['degree', 'cluster', 'orbit', 'spectral'] 
    kernels = {'degree':gaussian_emd, 
                'cluster':gaussian_emd, 
                'orbit':gaussian,
                'spectral':gaussian_emd}
    return methods, kernels


def load_control_net(model_x, model_adj, config):
    

    def get_node_name(name, parent_name):
        if len(name) <= len(parent_name):
            return False, ''
        p = name[:len(parent_name)]
        if p != parent_name:
            return False, ''
        return True, name[len(parent_name):]
        
    # import pdb;pdb.set_trace()
    
    pretrained_weights = torch.load(config.train.pretrain)
    
    pretrained_weights_x = pretrained_weights['x_state_dict']
    pretrained_weights_adj = pretrained_weights['adj_state_dict']
    
    
    model_x_scratch_dict = model_x.state_dict()
    model_adj_scratch_dict = model_adj.state_dict()
    
    # import pdb;pdb.set_trace()
    
    target_dict_x = {}
    for k in model_x_scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
        else:
            copy_k = k
        if copy_k in pretrained_weights_x:
            target_dict_x[k] = pretrained_weights_x[copy_k].clone()
            target_dict_x[k].requires_grad = False
        else:
            target_dict_x[k] = model_x_scratch_dict[k].clone()
            target_dict_x[k].requires_grad = True
            print(f'These weights are newly added: {k}, requires grad is {target_dict_x[k].requires_grad}')

    model_x.load_state_dict(target_dict_x, strict=True)
    
    target_dict_adj = {}
    for k in model_adj_scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
        else:
            copy_k = k
        if copy_k in pretrained_weights_adj:
            target_dict_adj[k] = pretrained_weights_adj[copy_k].clone()
            target_dict_adj[k].requires_grad = False
        else:
            target_dict_adj[k] = model_adj_scratch_dict[k].clone()
            target_dict_adj[k].requires_grad = True
            print(f'These weights are newly added: {k}, requires grad is {target_dict_adj[k].requires_grad}')

    model_adj.load_state_dict(target_dict_adj, strict=True)
    
    
    return model_x, model_adj
    # torch.save(model.state_dict(), output_path)
    # print('Done.')
