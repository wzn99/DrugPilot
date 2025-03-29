import torch
from torch.nn import Parameter
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/lk/project/repaint/MolPaint')

from models.layers import DenseGCNConv, MLP
from utils.graph_utils import mask_adjs, mask_x, pow_tensor
from models.attention import AttentionLayer

import pdb
from re import X
from matplotlib.pyplot import xkcd
from sympy import xfield
import csv
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import data as DATA
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np
from collections import OrderedDict

from utils.mol_utils import gen_mol, mols_to_smiles, smiles_to_mols, load_smiles, canonicalize_smiles, mols_to_nx
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol
from utils.process_drug_response import *
from utils.frag_onehot import onehot_to_string

import clip

def num2english(num, PRECISION=1):

    num = str(round(num,PRECISION)).split('.')[1]
    
    while len(num)!=PRECISION:
        num = num + '0'

    L1 = ["zero","one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    word = ""
    for i in str(num):
        # pdb.set_trace()
        word= word+" "+L1[int(i)]
   
    return word

class MLP_(torch.nn.Module):
    def __init__(self,num_i,num_h,num_o):
        super(MLP_,self).__init__()
        
        self.linear1=torch.nn.Linear(num_i,num_h)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(num_h,num_h) #2个隐层
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(num_h,num_o)
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
    
class BaselineNetworkLayer(torch.nn.Module):
    def __init__(self, num_linears, conv_input_dim, conv_output_dim, input_dim, output_dim, batch_norm=False):
        super(BaselineNetworkLayer, self).__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.convs.append(DenseGCNConv(conv_input_dim, conv_output_dim))
        self.hidden_dim = max(input_dim, output_dim)
        self.mlp_in_dim = input_dim + 2 * conv_output_dim
        self.mlp = MLP(num_linears, self.mlp_in_dim, self.hidden_dim, output_dim, 
                            use_bn=False, activate_func=F.elu)
        self.multi_channel = MLP(2, input_dim * conv_output_dim, self.hidden_dim, conv_output_dim, 
                                    use_bn=False, activate_func=F.elu)
        
    def forward(self, x, adj, flags):
        x_list = []
        for _ in range(len(self.convs)):
            _x = self.convs[_](x, adj[:, _, :, :])
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = torch.tanh(x_out)

        x_matrix = node_feature_to_matrix(x_out)
        mlp_in = torch.cat([x_matrix, adj.permute(0, 2, 3, 1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0, 3, 1, 2)
        _adj = _adj + _adj.transpose(-1, -2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out

class BaselineNetwork(torch.nn.Module):
    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
        super(BaselineNetwork, self).__init__()

        self.nfeat = max_feat_num
        self.max_node_num = max_node_num
        self.nhid = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nfeat, self.nhid, self.c_init, self.c_hid))
            elif _ == self.num_layers - 1:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_final))
            else:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_hid))

        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2 * self.fdim, output_dim=1, 
                            use_bn=False, activate_func=F.elu)
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)

    def forward(self, x, adj, flags=None):
        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):
            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)

        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        out_shape = adjs.shape[:-1]  # B x N x N
        score = self.final(adjs).view(*out_shape)

        self.mask = self.mask.to(score.device)
        score = score * self.mask
        score = mask_adjs(score, flags)

        return score

class ScoreNetworkA(BaselineNetwork):
    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
        super(ScoreNetworkA, self).__init__(max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                                            c_init, c_hid, c_final, adim, num_heads, conv)

        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(AttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init, 
                                                    self.c_hid, self.num_heads, self.conv))
            elif _ == self.num_layers - 1:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_final, self.num_heads, self.conv))
            else:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_hid, self.num_heads, self.conv))

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2 * self.fdim, output_dim=1, 
                            use_bn=False, activate_func=F.elu)
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)
        
    # def forward(self, x, adj, flags):
    #     adjc = pow_tensor(adj, self.c_init)

        
    #     # import pdb;pdb.set_trace()
        
    #     # adj_list = [adjc]
    #     for _ in range(self.num_layers):
    #         x, adjc = self.layers[_](x, adjc, flags)
    #         # adj_list.append(adjc)

    #     # import pdb;pdb.set_trace()
        
    #     # adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
    #     # del adj_list
    #     # out_shape = adjs.shape[:-1]  # B x N x N
    #     # score = self.final(adjs).view(*out_shape)
    #     self.mask = self.mask.to(x.device)
    #     score = adjc * self.mask
    #     score = mask_adjs(score, flags)
    #     return score.squeeze(1)
    
    def forward(self, x, adj, flags):
        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):
            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)

        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        del adj_list
        out_shape = adjs.shape[:-1]  # B x N x N
        score = self.final(adjs).view(*out_shape)
        self.mask = self.mask.to(score.device)
        score = score * self.mask
        score = mask_adjs(score, flags)
        return score

class ScoreNetworkX(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid):
        super(ScoreNetworkX, self).__init__()

        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        self.fdim = self.nfeat + self.depth * (self.nhid)
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2 * self.fdim, output_dim=self.nfeat, 
                            use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    # def forward(self, x, adj, flags):
    #     for _ in range(self.depth):
    #         x = self.layers[_](x, adj)
    #         x = self.activation(x)

    #     out_shape = (adj.shape[0], adj.shape[1], -1)
        

    #     x = self.final(x).view(*out_shape)

    #     x = mask_x(x, flags)

    #     return x

    def forward(self, x, adj, flags):
        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1)  # B x N x (F + num_layers x H)
        del x_list
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)

        x = mask_x(x, flags)

        return x

def save_cell_mut_matrix(cell_csv_path):
    f = open(cell_csv_path)
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        cell_id = item[1]
        mut = item[5]
        is_mutated = int(item[6])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))
    
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    return cell_dict, cell_feature

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
   
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class Cell(nn.Module):
    def __init__(self,
                 input_cell_feature_dim,
                 module_name,
                 fc_1_dim,
                 layer_num,
                 dropout,
                 layer_hyperparameter):
        super(Cell, self).__init__()

        self.module_name = module_name

        assert len(
            layer_hyperparameter) == layer_num, 'Number of layer is not same as hyperparameter list.'

        self.backbone = nn.Sequential()

        if module_name == "Transformer":

            for index, head in enumerate(layer_hyperparameter):
                transformer_encode_layer = nn.TransformerEncoderLayer(
                    d_model=input_cell_feature_dim, nhead=head, dropout=dropout)
                self.backbone.add_module('Transformer-{0}-{1}'.format(index, head), nn.TransformerEncoder(
                    transformer_encode_layer, 1))

            self.fc_1 = nn.Linear(input_cell_feature_dim, fc_1_dim)

        elif module_name == "Conv1d":
            input_channle = 1
            cell_feature_dim = input_cell_feature_dim

            for index, channel in enumerate(layer_hyperparameter['cnn_channels']):

                self.backbone.add_module('CNN1d-{0}_{1}_{2}'.format(index, input_channle, channel), nn.Conv1d(in_channels=input_channle,
                                                                                                              out_channels=channel,
                                                                                                              kernel_size=layer_hyperparameter['kernel_size'][index]))
                self.backbone.add_module('ReLU-{0}'.format(index), nn.ReLU())
                self.backbone.add_module('Maxpool-{0}'.format(index), nn.MaxPool1d(
                    layer_hyperparameter['maxpool1d'][index]))

                input_channle = channel
                cell_feature_dim = int(((
                    cell_feature_dim-layer_hyperparameter['kernel_size'][index]) + 1)/layer_hyperparameter['maxpool1d'][index])

            self.fc_1 = nn.Linear(cell_feature_dim*channel, fc_1_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.fc_1(x)
        return x


class Fusion(nn.Module):
    def __init__(self,
                 input_dim,
                 fc_1_dim,
                 fc_2_dim,
                 fc_3_dim,
                 dropout,
                 transformer_dropout,
                 fusion_mode):
        super(Fusion, self).__init__()

        self.fusion_mode = fusion_mode

        if fusion_mode == "concat":
            input_dim = input_dim[0]+input_dim[1]

            transformer_encode = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=1, dropout=transformer_dropout)

            self.transformer_layer = nn.TransformerEncoder(
                transformer_encode, 1)

            self.fc1 = nn.Linear(input_dim, fc_1_dim)

        self.fc2 = nn.Linear(fc_1_dim, fc_2_dim)
        self.fc25 = nn.Linear(fc_2_dim, 128)
        self.fc3 = nn.Linear(fc_2_dim, fc_3_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug, cell):

        if self.fusion_mode == "concat":
            x = torch.cat((drug, cell), 1)

        x = torch.unsqueeze(x, 1)
        x = self.transformer_layer(x)
        x = torch.squeeze(x, 1)

        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x_fusion_feature = self.relu(x)
        
        x = self.dropout(x_fusion_feature)
        x_number_feature = self.fc25(x)
        x_number_feature = self.relu(x_number_feature)
        x = self.fc3(x)
 
        x = nn.Sigmoid()(x)
        return  x, x_fusion_feature, x_number_feature


class CombinedScoreNetwork(torch.nn.Module):
    def __init__(self, device, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
        super(CombinedScoreNetwork, self).__init__()
        
        self.device = device
                
        self.score_network_x = ScoreNetworkX(max_feat_num=max_feat_num, depth=num_layers, nhid=nhid)
        self.score_network_adj = ScoreNetworkA(max_feat_num=max_feat_num, max_node_num=max_node_num, nhid=nhid, 
                                               num_layers=num_layers, num_linears=num_linears, c_init=c_init, 
                                               c_hid=c_hid, c_final=c_final, adim=adim, num_heads=num_heads, conv=conv)
        

        
        vocab_size = 50000
        transformer_width = 256
        # self.context_length = context_length
        self.context_length = 300
        self.context_num_length = 100
        transformer_width = 128
        transformer_layers = 3
        transformer_heads = 4
        embed_dim = 128
        
        # test encode
        self.token_embedding = nn.Embedding(vocab_size, transformer_width).to(self.device)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        
        self.token_embedding_num = nn.Embedding(vocab_size, transformer_width).to(self.device)
        self.positional_embedding_num = nn.Parameter(torch.empty(self.context_num_length, transformer_width))
        self.ln_final_num = LayerNorm(transformer_width)
        
        self.transformer_num = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask_num()
        )

        self.text_projection_num = nn.Parameter(torch.empty(transformer_width, embed_dim))


        
        
        self.drug_dim = 100
        self.drug_x_fc = torch.nn.Linear(self.drug_dim*10, 128)
        self.drug_adj_fc1 = torch.nn.Linear(self.drug_dim, 16)
        self.drug_adj_fc2 = torch.nn.Linear(self.drug_dim*16, 128)
        
        self.drug_fusion_fc_1 = torch.nn.Linear(110, 16)
        self.drug_fusion_fc_2 = torch.nn.Linear(100, 8)
        
        self.relu = torch.nn.ReLU()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        
        self.transformer_fusion = MLP_(embed_dim*2, 512, embed_dim*2)


        self.relu = nn.ReLU()
        
        # self.cell_module
        self.init_cell_module()

        # self.fusion_module
        self.init_fusion_module()
        

        
        
        
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask.to(self.device)
    
    def build_attention_mask_num(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_num_length, self.context_num_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask.to(self.device)
    
    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            
        nn.init.normal_(self.token_embedding_num.weight, std=0.02)
        nn.init.normal_(self.positional_embedding_num, std=0.01)
        proj_std = (self.transformer_num.width ** -0.5) * ((2 * self.transformer_num.layers) ** -0.5)
        attn_std = self.transformer_num.width ** -0.5
        fc_std = (2 * self.transformer_num.width) ** -0.5
        for block in self.transformer_num.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection_num, std=self.transformer_num.width ** -0.5)
  
            
            
    def encode_text(self, text):
        
        # pdb.set_trace()
        x = self.token_embedding(text.unsqueeze(2)).squeeze(2)  # [batch_size, n_ctx, d_model]
        

        
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
            
    def encode_num(self, text):
        
        # pdb.set_trace()
        x = self.token_embedding_num(text.unsqueeze(2)).squeeze(2)  # [batch_size, n_ctx, d_model]
        

        
        x = x + self.positional_embedding_num
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_num(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final_num(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection_num

        return x
    
    
    def init_cell_module(self):
        input_cell_feature_dim = 735
        module_name = 'Transformer'
        fc_1_dim = 128
        layer_num = 3
        dropout = 0.5
        layer_hyperparameter = [15,15,15]

        self.cell_module = Cell(input_cell_feature_dim,
                                module_name,
                                fc_1_dim,
                                layer_num,
                                dropout,
                                layer_hyperparameter)

    def init_fusion_module(self):
        input_dim = [128,128]
        fc_1_dim = 1024
        fc_2_dim = 256
        fc_3_dim = 1
        dropout = 0.5
        transformer_dropout = 0.5
        fusion_mode = 'concat'

        self.fusion_module = Fusion(input_dim,
                                    fc_1_dim,
                                    fc_2_dim,
                                    fc_3_dim,
                                    dropout,
                                    transformer_dropout,
                                    fusion_mode)
            
    def get_cell_matrix(self, path):  
        self.cell_matrix = save_cell_mut_matrix(path)


    def generate_samples(self, start, end):
        # data: list, consist of [drug smile, cell line, ic50]
        descriptions_number = []
        # assert end - start == 1000
        
        # if model.training:    
        for ic50 in range(start,end,1):
        # for idx, ic50 in enumerate(range(0,1000,1)):
            des = "zero point " + num2english(ic50/1000, PRECISION=3)
            descriptions_number.append(des)

        text = clip.tokenize(descriptions_number, context_length = 100).to(self.device)
        # pdb.set_trace()
        text_features = self.encode_num(text)
        del text, descriptions_number
        torch.cuda.empty_cache()
        return text_features
    
    def forward(self, data):
        adj = data.edge_index.permute(2, 0, 1)
        x = data.x
        cell_name = data.cell_name
        
        
        device = x.device
        
        x_drug = self.score_network_x(x, adj.float(), None)
        adj_drug = self.score_network_adj(x, adj.float(), None)
        
        target = [torch.FloatTensor(self.cell_matrix[1][self.cell_matrix[0][str(int(i))]]).reshape(1,1,735) for i in cell_name]
        target = torch.cat(target,dim=0).to(device)
        
        cell_feature = self.cell_module(target)
        
        # import pdb;pdb.set_trace()
        drug_feature = self.relu(self.drug_fusion_fc_1(torch.cat([x_drug, adj_drug], dim=2)))
        drug_feature = self.relu(self.drug_fusion_fc_2(drug_feature.permute(0,2,1)))
        drug_feature = drug_feature.reshape(x.shape[0],-1)
        
        y_pred, fusion_features, fusion_number_features = self.fusion_module(drug_feature, cell_feature)

        
        descriptions_text = []
        descriptions_number = []
        for index, item in enumerate(data.ic50):
            # pdb.set_trace()
            
            # des =  str(translateNumberToEnglish(item.item()))
            
            des = "zero point " + num2english(item.item(), PRECISION=3)
            descriptions_number.append(des)
            
            des = "The drug response value with " + str(cell_name[index].int().item()) +" is "
            descriptions_text.append(des)
            
            # continue
        
        text = clip.tokenize(descriptions_text,context_length=300).to(device)
        number = clip.tokenize(descriptions_number,context_length=100).to(device)
        # 
      
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
        del text, number
        torch.cuda.empty_cache()

        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        # normalized features
        number_features = number_features / number_features.norm(dim=1, keepdim=True)
        fusion_number_features = fusion_number_features / fusion_number_features.norm(dim=1, keepdim=True)
        fusion_features = fusion_features / fusion_features.norm(dim=1, keepdim=True)
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        # pdb.set_trace()
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_dc = logit_scale * fusion_features @ sentence_features.t()
        # logits_per_dc =  fusion_features @ text_features.t()
        logits_per_text = logits_per_dc.t()


        num_logits_per_dc = logit_scale * fusion_number_features @ number_features.t()
        # logits_per_dc =  fusion_features @ text_features.t()
        num_logits_per_text = logits_per_dc.t()
        
        # shape = [global_batch_size, global_batch_size]
        return  logits_per_dc, logits_per_text, num_logits_per_dc, num_logits_per_text, y_pred
    
    

