import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool
import clip
from collections import OrderedDict


class GNN(nn.Module):
    def __init__(self,
                 input,
                 output,
                 gnn_type,
                 heads=1,
                 dropout=0.2,
                 feature_pre_dropout=0,
                 activate_func='relu'):
        super(GNN, self).__init__()

        self.gnn_type = gnn_type

        if feature_pre_dropout>0:
            self.pre_dropout = nn.Dropout(feature_pre_dropout)

        if self.gnn_type == 'GINConvNet':
            nn_core = Sequential(Linear(input, output),
                                 ReLU(), Linear(output, output))
            self.gnn = GINConv(nn_core)
            self.bn = torch.nn.BatchNorm1d(output)
        elif self.gnn_type == 'GCNConv':
            self.gnn = GCNConv(input, output)
        elif self.gnn_type == 'GATConv':
            self.gnn = GATConv(input, output, heads=heads, dropout=dropout)

        if activate_func == 'relu':
            self.activate_func = nn.ReLU()
        elif activate_func == 'elu':
            self.activate_func = nn.ELU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if hasattr(self, 'pre_dropout'):
            x = self.pre_dropout(x)

        if self.gnn_type == 'GINConvNet':
            x = self.gnn(x, edge_index)
            x = self.activate_func(x)
            x = self.bn(x)
        elif self.gnn_type == 'GCNConv':
            x = self.gnn(x, edge_index)
            x = self.activate_func(x)
        elif self.gnn_type == 'GATConv':
            x = self.gnn(x, edge_index)
            x = self.activate_func(x)

        data.x = x

        return data


class Drug(nn.Module):
    def __init__(self,
                 module_name,
                 input_drug_feature_dim,
                 output_drug_feature_dim,
                 layer_num,
                 graph_pooling,
                 linear_layers,
                 gnn_layers,
                 dropout):
        super(Drug, self).__init__()

        assert len(
            gnn_layers) == layer_num, 'Number of layer is not same as hyperparameter list.'
        assert graph_pooling in [
            'add', 'max', 'mean', 'max_mean'], 'The type of graph pooling is not right.'

        self.gnn_layers = gnn_layers
        self.linear_layers = linear_layers
        self.graph_pooling = graph_pooling
        self.backbone = nn.Sequential()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # self.fc = nn.Linear(output_drug_feature_dim, num_labels)

        for index, params in enumerate(gnn_layers):
            if module_name[index] == "GATConv":
                self.backbone.add_module(
                    '{0}-{1}'.format(module_name[index], index), GNN(params['intput'], params['output'], module_name[index], heads=params['heads'], dropout=params['dropout'], feature_pre_dropout=params['feature_pre_dropout']))
            else:
                self.backbone.add_module(
                    '{0}-{1}'.format(module_name[index], index), GNN(params['intput'], params['output'], module_name[index]))

        if linear_layers:
            self.linears = nn.Sequential()

            for idx, linear_parameter in enumerate(linear_layers):

                if linear_parameter['operate_name'] == 'linear':
                    self.linears.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), torch.nn.Linear(linear_parameter['param'][0], linear_parameter['param'][1]))

                elif linear_parameter['operate_name'] == 'relu':
                    self.linears.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), self.relu)

                elif linear_parameter['operate_name'] == 'dropout':
                    self.linears.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), nn.Dropout(linear_parameter['param']))

    def forward(self, data):

        data = self.backbone(data)
        x, batch = data.x, data.batch

        if self.graph_pooling == "add":
            x = global_add_pool(x, batch)
        if self.graph_pooling == "max":
            x = gmp(x, batch)
        if self.graph_pooling == "mean":
            x = gap(x, batch)
        if self.graph_pooling == "max_mean":
            x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        drug_feature = self.linears(x)

        return drug_feature



class GraphMultiHead(torch.nn.Module):
    def __init__(self, config):
        super(GraphMultiHead, self).__init__()
        self.config = config

        # self.drug_module
        self.init_drug_module(self.config['model']['drug_module'])
        
        self.task_heads = nn.ModuleList([
    nn.Linear(config['model']['drug_module']['output_drug_feature_dim'], output_dim)
    for output_dim in config['task_output_dims']  # `task_output_dims` is a list of the number of labels for each task
])
        self.selected_task = config['selected_task'] if config.get('selected_task') is not None else None
        self.transfer_task_head = nn.Linear(config['model']['drug_module']['output_drug_feature_dim'], config['transfer_task_dim']) if config.get('transfer_task_dim') is not None else None
        
        self.loss_fn = nn.MSELoss()
        self.loss_fn_l1 = nn.L1Loss(reduction='mean')

        self.loss_domain = torch.nn.NLLLoss()

        self.memo = {}

        self.logit_scale = 0.1


    def init_drug_module(self, config):
        module_name = config['module_name']
        input_drug_feature_dim = config['input_drug_feature_dim']
        layer_num = config['layer_num']
        graph_pooling = config['graph_pooling']
        dropout = config['dropout']
        output_drug_feature_dim = config['output_drug_feature_dim']
        linear_layers = config['linear_layers'] if config.get(
            'linear_layers') else None
        gnn_layers = config['gnn_layers']


        self.drug_module = Drug(module_name,
                                input_drug_feature_dim,
                                output_drug_feature_dim,
                                layer_num,
                                graph_pooling,
                                linear_layers,
                                gnn_layers,
                                dropout)
        

    def forward(self, data):

        device = data.x.device
        batch = data.batch

        drug_features = self.drug_module(data)
        if self.selected_task is not None:
            if self.selected_task < 9:
                selected_task_predictions = self.task_heads[self.selected_task](drug_features)
                return selected_task_predictions
            else:
                transfer_task_predictions = self.transfer_task_head(drug_features)
                return transfer_task_predictions

        task_predictions = [task_head(drug_features) for task_head in self.task_heads]
        return task_predictions
