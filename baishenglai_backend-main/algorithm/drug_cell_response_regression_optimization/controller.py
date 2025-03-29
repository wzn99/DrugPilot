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

from models import *
from models.ScoreNetwork_A import ScoreNetworkA
from models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH


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


class Drug(nn.Module):
    def __init__(self,
                 input_drug_feature_dim,
                 use_drug_edge,
                 input_drug_edge_dim,
                 fc_1_dim,
                 fc_2_dim,
                 dropout,
                 transformer_dropout,
                 show_attenion=False):
        super(Drug, self).__init__()

        self.use_drug_edge = use_drug_edge
        self.show_attenion = show_attenion
        if use_drug_edge:
            self.gnn1 = GATConv(
                input_drug_feature_dim, input_drug_feature_dim, heads=10, edge_dim=input_drug_feature_dim)
            self.edge_embed = torch.nn.Linear(
                input_drug_edge_dim, input_drug_feature_dim)
        else:
            self.gnn1 = GATConv(input_drug_feature_dim,
                                input_drug_feature_dim, heads=10)

        self.trans_layer_encode_1 = nn.TransformerEncoderLayer(
            d_model=input_drug_feature_dim, nhead=1, dropout=transformer_dropout)
        self.trans_layer_1 = nn.TransformerEncoder(
            self.trans_layer_encode_1, 1)

        self.trans_layer_encode_2 = nn.TransformerEncoderLayer(
            d_model=input_drug_feature_dim*10, nhead=1, dropout=transformer_dropout)
        self.trans_layer_2 = nn.TransformerEncoder(
            self.trans_layer_encode_2, 1)

        self.gnn2 = GCNConv(input_drug_feature_dim*10,
                            input_drug_feature_dim*10)

        self.fc_1 = torch.nn.Linear(input_drug_feature_dim*10*2, fc_1_dim)
        self.fc_2 = torch.nn.Linear(fc_1_dim, fc_2_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        if self.use_drug_edge:
            x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
            edge_embeddings = self.edge_embed(edge_attr.float())
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        x = torch.unsqueeze(x, 1)
        x = self.trans_layer_1(x)
        x = torch.squeeze(x, 1)

        if self.use_drug_edge:
            x = self.gnn1(x, edge_index, edge_attr=edge_embeddings)
        else:
            x = self.gnn1(x, edge_index)

        x = self.relu(x)

        x = torch.unsqueeze(x, 1)
        x = self.trans_layer_2(x)
        x = torch.squeeze(x, 1)

        x = self.gnn2(x, edge_index)
        x = self.relu(x)

        if self.show_attenion:
            self.show_atom_attention(x, data)

        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)

        return x
    
    def show_atom_attention(self, x, data):
        x_heat = torch.sum(x, 1)

        from rdkit.Chem import Draw
        from rdkit import Chem
        from tqdm import tqdm
        import numpy as np

        for index, i in enumerate(tqdm(data.smiles)):
            if index >= 50:
                break
            m = Chem.MolFromSmiles(i)
            for atom in m.GetAtoms():
                atom.SetProp("atomNote", str(atom.GetIdx()))

            from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
            opts = DrawingOptions()

            opts.includeAtomNumbers = True
            opts.bondLineWidth = 2.8
            draw = Draw.MolToImage(m, size=(600, 600), options=opts)

            smile_name = i.replace('\\', '!').replace('/', '~')

            draw.save('./infer/img/{}.jpg'.format(smile_name))

            heat_item = x_heat.numpy()[np.argwhere(
                data.batch.numpy() == index)]

            with open('./infer/heat/{}.txt'.format(smile_name), 'w') as f:
                for idx, heat in enumerate(heat_item):
                    f.write(str(heat[0])+'\t'+str(idx)+'\n')


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

class MLP(torch.nn.Module):
    
 
    def __init__(self,num_i,num_h,num_o):
        super(MLP,self).__init__()
        
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
    
class TransEDRP(torch.nn.Module):
    def __init__(self, config, config_model_x, config_model_adj, device, train_flags):
        super(TransEDRP, self).__init__()
        clip.available_models()

        self.config = config
        self.config_model_x = config_model_x
        self.config_model_adj = config_model_adj
        self.device = device

        self.train_flags = train_flags
        
        # self.drug_module
        self.init_drug_module()

        # self.cell_module
        self.init_cell_module(self.config['model']['cell_module'])

        # self.fusion_module
        self.init_fusion_module(self.config['model'])

        
  
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
        self.drug_fusion_fc = torch.nn.Linear(256, 128)
        self.relu = torch.nn.ReLU()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        
        self.transformer_fusion = MLP(embed_dim*2, 512, embed_dim*2)

        
    def get_cell_matrix(self, path):  
        self.cell_matrix = save_cell_mut_matrix(path)

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
  
            
    def init_drug_module(self):
        max_feat_num, depth, nhid = self.config_model_x['max_feat_num'], self.config_model_x['depth'], self.config_model_x['nhid']
        self.drug_x_module = ScoreNetworkX(max_feat_num, depth, nhid)
        
        max_feat_num, max_node_num, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads, conv =  self.config_model_adj['max_feat_num'], self.config_model_adj['max_node_num'], self.config_model_adj['nhid'], self.config_model_adj['num_layers'], self.config_model_adj['num_linears'], self.config_model_adj['c_init'], self.config_model_adj['c_hid'], self.config_model_adj['c_final'], self.config_model_adj['adim'], self.config_model_adj['num_heads'], self.config_model_adj['conv']
        self.drug_adj_module = ScoreNetworkA(max_feat_num, max_node_num, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads, conv)
        
    def init_cell_module(self, config):
        input_cell_feature_dim = config['input_cell_feature_dim']
        module_name = config['module_name']
        fc_1_dim = config['fc_1_dim']
        layer_num = config['layer_num']
        dropout = config['transformer_dropout'] if config.get(
            'transformer_dropout') else 0
        layer_hyperparameter = config['layer_hyperparameter']

        self.cell_module = Cell(input_cell_feature_dim,
                                module_name,
                                fc_1_dim,
                                layer_num,
                                dropout,
                                layer_hyperparameter)
        
    def init_fusion_module(self, config):
        input_dim = [config['drug_module']['fc_2_dim'],
                     config['cell_module']['fc_1_dim']]
        fc_1_dim = config['fusion_module']['fc_1_dim']
        fc_2_dim = config['fusion_module']['fc_2_dim']
        fc_3_dim = config['fusion_module']['fc_3_dim']
        dropout = config['fusion_module']['dropout']
        transformer_dropout = config['fusion_module']['transformer_dropout']
        fusion_mode = config['fusion_module']['fusion_mode']

        self.fusion_module = Fusion(input_dim,
                                    fc_1_dim,
                                    fc_2_dim,
                                    fc_3_dim,
                                    dropout,
                                    transformer_dropout,
                                    fusion_mode)

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
    
    
    def forward(self, data):
        
        device = data.x.device
        x_drug = self.drug_module(data)
        x_cell = self.cell_module(data.target[:, None, :])
        pred_y, fusion_features = self.fusion_module(x_drug, x_cell)
        
        
        descriptions_text = []
        descriptions_number = []
        for index, item in enumerate(data.y):
            # pdb.set_trace()
            
            # des =  str(translateNumberToEnglish(item.item()))
            
            des = "zero point " + num2english(item.item(), PRECISION=3)
            descriptions_number.append(des)
            
            des = "The drug response value between " + data.smiles[index] + " and "+ data.cell_name[index] +" is "
            descriptions_text.append(des)
            
            # continue
        
        text = clip.tokenize(descriptions_text,context_length=300).to(device)
        number = clip.tokenize(descriptions_number,context_length=100).to(device)
        # 
      
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        # normalized features
        fusion_features = fusion_features / fusion_features.norm(dim=1, keepdim=True)
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        # pdb.set_trace()
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_dc = logit_scale * fusion_features @ sentence_features.t()
        # logits_per_dc =  fusion_features @ text_features.t()
        logits_per_text = logits_per_dc.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_dc, logits_per_text
    
    
        # return  fusion_features, text_features
            
            
            # reg_loss = self.loss_fn(pred_y.float(), data.y.view(-1, 1).float().to(device))
            
            # rank_loss = self.ranking_loss(data.y.view(-1, 1).float().to(device) , pred_y.float(), torch.ones_like(pred_y.float()))

            # main_loss = reg_loss * 0.9 + rank_loss * 0.1

            # return main_loss, fusion_feature
            
            
    def forward_MSE(self, x, adj, cell_name, ic50):
            
        device = x.device
        # pdb.set_trace()
        x_drug = self.drug_x_module(x, adj, None)
        adj_drug = self.drug_adj_module(x, adj, None)
        
        target = [torch.FloatTensor(self.cell_matrix[1][self.cell_matrix[0][str(int(i))]]).reshape(1,1,735) for i in cell_name]
        target = torch.cat(target,dim=0).to(device)
        
        cell_feature = self.cell_module(target)
        
        x_drug = x_drug.reshape(-1, x_drug.shape[1] * x_drug.shape[2])
        x_drug = self.relu(self.drug_x_fc(x_drug))
        
        # pdb.set_trace()
        adj_drug = self.relu(self.drug_adj_fc1(adj_drug))
        adj_drug = adj_drug.reshape(-1, adj_drug.shape[1] * adj_drug.shape[2])
        adj_drug = self.relu(self.drug_adj_fc2(adj_drug))
        
        drug_feature = self.relu(self.drug_fusion_fc(torch.cat([x_drug, adj_drug], dim=1)))
        
        pred_y, _, _ = self.fusion_module(drug_feature, cell_feature)
        
        return pred_y
    
 
    def forward4gen(self, x, adj, cell_name, ic50):
            
        device = x.device
        # pdb.set_trace()
        with torch.no_grad():
            x_drug = self.drug_x_module(x, adj, None)
            adj_drug = self.drug_adj_module(x, adj, None)
            
        target = [torch.FloatTensor(self.cell_matrix[1][self.cell_matrix[0][str(int(i))]]).reshape(1,1,735) for i in cell_name]
        target = torch.cat(target,dim=0).to(device)
        
        cell_feature = self.cell_module(target)
        
        x_drug = x_drug.reshape(-1, x_drug.shape[1] * x_drug.shape[2])
        x_drug = self.relu(self.drug_x_fc(x_drug))
        
        # pdb.set_trace()
        adj_drug = self.relu(self.drug_adj_fc1(adj_drug))
        adj_drug = adj_drug.reshape(-1, adj_drug.shape[1] * adj_drug.shape[2])
        adj_drug = self.relu(self.drug_adj_fc2(adj_drug))
        
        drug_feature = self.relu(self.drug_fusion_fc(torch.cat([x_drug, adj_drug], dim=1)))
        
     
        _, fusion_features, fusion_number_features = self.fusion_module(drug_feature, cell_feature)
        
        # pdb.set_trace()
        descriptions_text = []
        descriptions_number = []
        for index, item in enumerate(ic50):
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
        return  logits_per_dc, logits_per_text, num_logits_per_dc, num_logits_per_text
    
    
        # return  fusion_features, text_features
            
            
            # reg_loss = self.loss_fn(pred_y.float(), data.y.view(-1, 1).float().to(device))
            
            # rank_loss = self.ranking_loss(data.y.view(-1, 1).float().to(device) , pred_y.float(), torch.ones_like(pred_y.float()))

            # main_loss = reg_loss * 0.9 + rank_loss * 0.1

            # return main_loss, fusion_feature
            
    def infer(self, data):
        device = data.x.device
        x_drug = self.drug_module(data)
        x_cell = self.cell_module(data.target[:, None, :])
        pred_y, fusion_features = self.fusion_module(x_drug, x_cell)
        fusion_features = fusion_features / fusion_features.norm(dim=1, keepdim=True)
        return fusion_features
    
    def get_text_embed(self, x, adj, labels, config):
        
        device = x.device
  
    
        descriptions_text = []
        descriptions_number = []

        
        for idx, ic50 in enumerate(range(0,10,1)):
            
            des = "zero point " + num2english(ic50/10)
            descriptions_number.append(des)
            
            
            # des = "The drug response value between " + item + " and "+ str(labels['cell']) +" is "
            des = "The drug response value with " + str(labels['cell']) +" is "
            descriptions_text.append(des)
                
            
        text = clip.tokenize(descriptions_text,context_length=300).to(device)
        number = clip.tokenize(descriptions_number,context_length=100).to(device)
      
        
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        pdb.set_trace()
        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        # normalized features
        # fusion_features = fusion_features / fusion_features.norm(dim=1, keepdim=True)
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)
        number_features = number_features / number_features.norm(dim=1, keepdim=True)
        return sentence_features, number_features
    
    def GaussianFourierProjection(self, ve_noise_scale):
        device = ve_noise_scale.device
        self.noise_embed = 128
        self.ve_noise_embed = nn.Parameter(torch.rand(self.noise_embed)).to(device)
        nn.init.normal_(self.ve_noise_embed, std=16)
        
        self.noise_fc_1 = torch.nn.Linear(256, 128).to(device)
        self.noise_fc_2 = torch.nn.Linear(256, 128).to(device)
        self.noise_fc_3 = torch.nn.Linear(256, 256).to(device)
        self.noise_fc_4 = torch.nn.Linear(256, 128).to(device)
        
        ve_noise_embed = ve_noise_scale[:, None] *  self.ve_noise_embed[None, :].to(device) * 2 * 3.141592653589793
        return  torch.cat([torch.sin(ve_noise_embed), torch.cos(ve_noise_embed)], axis=-1)
    
    
    def swish(self, x):
        return torch.sigmoid(x) * x
    
    def get_drug_embed(self, x, adj, ve_noise_scale, flag_noise):

        device = x.device
        # 
        # print("1.1:{}".format(torch.cuda.max_memory_reserved(0)))
        x_drug = self.drug_x_module(x, adj, None)
        # print("1.2:{}".format(torch.cuda.max_memory_reserved(0)))
        adj_drug = self.drug_adj_module(x, adj, None)
        # print("1.3:{}".format(torch.cuda.max_memory_reserved(0)))
        # print("1.4:{}".format(torch.cuda.max_memory_reserved(0)))
        x_drug = x_drug.reshape(-1, x_drug.shape[1] * x_drug.shape[2])
        
        # pdb.set_trace()
        if flag_noise:
            temb = self.GaussianFourierProjection(torch.log(ve_noise_scale))
            temb = self.swish(temb)
            
            x_drug = self.relu(self.drug_x_fc(x_drug))
            x_drug = x_drug+self.swish(self.noise_fc_1(temb))
            
            adj_drug = self.relu(self.drug_adj_fc1(adj_drug))
            adj_drug = adj_drug.reshape(-1, adj_drug.shape[1] * adj_drug.shape[2])
            adj_drug = self.relu(self.drug_adj_fc2(adj_drug))   
            adj_drug = adj_drug+self.swish(self.noise_fc_2(temb))
            
            drug_feature = torch.cat([x_drug, adj_drug], dim=1)
            drug_feature = drug_feature+self.swish(self.noise_fc_3(temb))
            drug_feature = self.relu(self.drug_fusion_fc(drug_feature))
            drug_feature = drug_feature+self.swish(self.noise_fc_4(temb))
            
        else:
            x_drug = self.relu(self.drug_x_fc(x_drug))
            adj_drug = self.relu(self.drug_adj_fc1(adj_drug))
            adj_drug = adj_drug.reshape(-1, adj_drug.shape[1] * adj_drug.shape[2])
            adj_drug = self.relu(self.drug_adj_fc2(adj_drug))   
            drug_feature = torch.cat([x_drug, adj_drug], dim=1)
            drug_feature = self.relu(self.drug_fusion_fc(drug_feature))
            
        return drug_feature
    
        
    def get_fusion_embed(self, x, adj, ve_noise_scale, labels):
        # pdb.set_trace()
        device = x.device
        # 
        # print("1.1:{}".format(torch.cuda.max_memory_reserved(0)))
        x_drug = self.drug_x_module(x, adj, None)
        # print("1.2:{}".format(torch.cuda.max_memory_reserved(0)))
        adj_drug = self.drug_adj_module(x, adj, None)
        # print("1.3:{}".format(torch.cuda.max_memory_reserved(0)))
        with torch.no_grad():
            target = torch.FloatTensor(self.cell_matrix[1][self.cell_matrix[0][str(int(labels['cell']))]]).reshape(1,1,735)
            target = target.to(device)
            target = target.repeat(x_drug.size()[0], 1, 1)
            cell_feature = self.cell_module(target)
        # print("1.4:{}".format(torch.cuda.max_memory_reserved(0)))
        x_drug = x_drug.reshape(-1, x_drug.shape[1] * x_drug.shape[2])
        
        
        temb = self.GaussianFourierProjection(torch.log(ve_noise_scale))
        
        temb = self.swish(temb)
        
        x_drug = self.relu(self.drug_x_fc(x_drug))
        
        x_drug = x_drug+self.swish(self.noise_fc_1(temb))
        
        
        
        adj_drug = self.relu(self.drug_adj_fc1(adj_drug))
        adj_drug = adj_drug.reshape(-1, adj_drug.shape[1] * adj_drug.shape[2])
        adj_drug = self.relu(self.drug_adj_fc2(adj_drug))   
        
        adj_drug = adj_drug+self.swish(self.noise_fc_2(temb))
        
        
        drug_feature = torch.cat([x_drug, adj_drug], dim=1)
 
        drug_feature = drug_feature+self.swish(self.noise_fc_3(temb))
        
        drug_feature = self.relu(self.drug_fusion_fc(drug_feature))
        

        drug_feature = drug_feature+self.swish(self.noise_fc_4(temb))
      
        pred_y, fusion_features, number_fusion_features = self.fusion_module(drug_feature, cell_feature)
        
        # normalized features
        fusion_features = fusion_features / fusion_features.norm(dim=1, keepdim=True)
        number_fusion_features = number_fusion_features / number_fusion_features.norm(dim=1, keepdim=True)
        return fusion_features, number_fusion_features
    
         # pdb.set_trace()

    
    def multi_data_text_forward(self, data, device=None):
        
        device = self.device
      
        descriptions_text = []
        descriptions_number = []
        for index, item in enumerate(data):
            if item[0][0] == 0 and item[0][1] == 0:
                descriptions_number.append("")
                descriptions_text.append("")
            else:
                des = "zero point" + num2english(item[0][1].item(), PRECISION=3)
                descriptions_number.append(des)
                
                des = "The drug response value about " + str(item[0][0].int().item()) +" is "
                descriptions_text.append(des)
            
        text = clip.tokenize(descriptions_text,context_length=300).to(device)
        number = clip.tokenize(descriptions_number,context_length=100).to(device)
        # 
      
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)
        # mask = self.create_mask(sentence_features.unsqueeze(-1),0.5)
        # sentence_features = sentence_features.unsqueeze(-1) * mask.unsqueeze(-1).expand_as(sentence_features.unsqueeze(-1)).float()
        
        # shape = [global_batch_size, global_batch_size]
        return sentence_features.unsqueeze(-1).float()
    
    
    def text_forward(self, data, device=None):
        
        
        
        device = self.device
      
        descriptions_text = []
        descriptions_number = []
        for index, item in enumerate(data):
            # des =  str(translateNumberToEnglish(item.item()))
            # import pdb;pdb.set_trace()
            des = "zero point" + num2english(item[0][1].item(), PRECISION=3)
            descriptions_number.append(des)
            
            des = "The drug response value about " + str(item[0][0].int().item()) +" is "
            descriptions_text.append(des)
            
        text = clip.tokenize(descriptions_text,context_length=300).to(device)
        number = clip.tokenize(descriptions_number,context_length=100).to(device)
        # 
      
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        mask = self.create_mask(sentence_features.unsqueeze(-1),0.5)
        
        sentence_features = sentence_features.unsqueeze(-1) * mask.unsqueeze(-1).expand_as(sentence_features.unsqueeze(-1)).float()
        # shape = [global_batch_size, global_batch_size]
        return sentence_features
    
    def text_full_forward(self, data, device=None):
        
        
        
        device = self.device
      
        descriptions_text = []
        descriptions_number = []
        for index, item in enumerate(data):
            # des =  str(translateNumberToEnglish(item.item()))
            # import pdb;pdb.set_trace()
            des = "zero point" + num2english(item[0][1].item(), PRECISION=3)
            descriptions_number.append(des)
            
            des = "The drug response value about " + str(item[0][0].int().item()) +" is "
            descriptions_text.append(des)
            
        text = clip.tokenize(descriptions_text,context_length=300).to(device)
        number = clip.tokenize(descriptions_number,context_length=100).to(device)
        # 

        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)
        # import pdb;pdb.set_trace()
        mask = self.create_mask(sentence_features.unsqueeze(-1),0)
        
        sentence_features = sentence_features.unsqueeze(-1) * mask.unsqueeze(-1).expand_as(sentence_features.unsqueeze(-1)).float()
        # shape = [global_batch_size, global_batch_size]
        return sentence_features
    
    def create_mask(self, embeddings, word_dropout):
        # word_dropout =0 then mask is not useful, mask will be ones matrix. 
        batch, length, size = embeddings.size()
        mask = embeddings.new_empty(batch, 1)
        mask = mask.bernoulli_(1 - word_dropout)
        # embeddings = embeddings * mask.unsqueeze(-1).expand_as(embeddings).float()
        return mask
    
    
        
    def forward_null_text(self, data, device=None):
        
        device = self.device

        
        descriptions_text = []
        descriptions_number = []
        for index, item in enumerate(data):
            
            
            # des =  str(translateNumberToEnglish(item.item()))
            # import pdb;pdb.set_trace()
            des = ""
            descriptions_number.append(des)
            
            des = ""
            descriptions_text.append(des)
            
            # continue
        
        text = clip.tokenize(descriptions_text,context_length=300).to(device)
        number = clip.tokenize(descriptions_number,context_length=100).to(device)
        # 
      
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        mask = self.create_mask(sentence_features.unsqueeze(-1),0)
        
        sentence_features = sentence_features.unsqueeze(-1) * mask.unsqueeze(-1).expand_as(sentence_features.unsqueeze(-1)).float()
        # shape = [global_batch_size, global_batch_size]
        
        return sentence_features
    
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
        
        return text_features
    
class TransFGDRP(torch.nn.Module):
    def __init__(self, config, config_model_x, config_model_adj, device, train_flags):
        super(TransFGDRP, self).__init__()
        clip.available_models()

        self.config = config
        self.config_model_x = config_model_x
        self.config_model_adj = config_model_adj
        self.device = device

        self.train_flags = train_flags
        
        # self.drug_module
        self.init_drug_module()
       
        vocab_size = 50000
        transformer_width = 256
        # self.context_length = context_length
        self.context_length = 512
        self.context_num_length = 100
        transformer_width = 128
        transformer_layers = 3
        transformer_heads = 4
        embed_dim = 128
        
        # text encode
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
                   
        self.drug_dim = 100
        self.drug_x_fc = torch.nn.Linear(self.drug_dim*10, 128)
        self.drug_adj_fc1 = torch.nn.Linear(self.drug_dim, 16)
        self.drug_adj_fc2 = torch.nn.Linear(self.drug_dim*16, 128)
        self.drug_fusion_fc = torch.nn.Linear(256, embed_dim*2)
        self.relu = torch.nn.ReLU()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        
        self.transformer_fusion = MLP(embed_dim, 256, embed_dim*2)
        
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
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
                       
    def init_drug_module(self):
        max_feat_num, depth, nhid = self.config_model_x['max_feat_num'], self.config_model_x['depth'], self.config_model_x['nhid']
        self.drug_x_module = ScoreNetworkX(max_feat_num, depth, nhid)
        
        max_feat_num, max_node_num, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads, conv =  self.config_model_adj['max_feat_num'], self.config_model_adj['max_node_num'], self.config_model_adj['nhid'], self.config_model_adj['num_layers'], self.config_model_adj['num_linears'], self.config_model_adj['c_init'], self.config_model_adj['c_hid'], self.config_model_adj['c_final'], self.config_model_adj['adim'], self.config_model_adj['num_heads'], self.config_model_adj['conv']
        self.drug_adj_module = ScoreNetworkA(max_feat_num, max_node_num, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads, conv)

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
 
    def forward2gen(self, x, adj, frag):
            
        device = x.device
        # pdb.set_trace()
        with torch.no_grad():
            x_drug = self.drug_x_module(x, adj, None)
            adj_drug = self.drug_adj_module(x, adj, None)
                  
        x_drug = x_drug.reshape(-1, x_drug.shape[1] * x_drug.shape[2])
        x_drug = self.relu(self.drug_x_fc(x_drug))
        
        # pdb.set_trace()
        adj_drug = self.relu(self.drug_adj_fc1(adj_drug))
        adj_drug = adj_drug.reshape(-1, adj_drug.shape[1] * adj_drug.shape[2])
        adj_drug = self.relu(self.drug_adj_fc2(adj_drug))
        
        drug_feature = self.relu(self.drug_fusion_fc(torch.cat([x_drug, adj_drug], dim=1)))
        
        # pdb.set_trace()
        descriptions_text = []
        for index, item in enumerate(frag):
            frags = onehot_to_string(item.cpu().numpy())
            if '?' in frags:
                frags = frags.replace('?', ', ')
            # pdb.set_trace()
            des = 'This drug contains ' + frags
            descriptions_text.append(des)
            
            # continue
        
        text = clip.tokenize(descriptions_text,context_length=self.context_length).to(device)
      
        text_features = self.encode_text(text)
        
        sentence_features = self.transformer_fusion(text_features)
        # normalized features
        drug_feature = drug_feature / drug_feature.norm(dim=1, keepdim=True)
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        # pdb.set_trace()
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_df = logit_scale * drug_feature @ sentence_features.t()
        logits_per_text = logits_per_df.t()
        
        # shape = [global_batch_size, global_batch_size]
        return  logits_per_df, logits_per_text
                 
    def text_forward(self, data, device=None):
        
        device = self.device
        
        descriptions_text = []
        

        x_percent = 20
        
        for index, item in enumerate(data):
            if random.randint(1, 100) <= x_percent:
                des = ''
            else:
                frags = onehot_to_string(item.cpu().numpy())
                if '?' in frags:
                    frags = frags.replace('?', ', ')
                # pdb.set_trace()
                des = 'This drug contains ' + frags
            descriptions_text.append(des)
            
            
        text = clip.tokenize(descriptions_text,context_length=self.context_length).to(device)

        text_features = self.encode_text(text)
        
        sentence_features = self.transformer_fusion(text_features)
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        # shape = [global_batch_size, global_batch_size]
        return sentence_features.unsqueeze(-1).float()
    
    
    def text_full_forward(self, data, device=None):
        
        device = self.device
      
        descriptions_text = []

        for index, frags in enumerate(data):
            # frags = onehot_to_string(item.cpu().numpy())
            if '?' in frags:
                frags = frags.replace('?', ', ')
            # pdb.set_trace()
            des = 'This drug contains ' + frags
            descriptions_text.append(des)
            
        text = clip.tokenize(descriptions_text,context_length=self.context_length).to(device)
        # 
      
        text_features = self.encode_text(text)
        
        sentence_features = self.transformer_fusion(text_features)
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)


        # sentence_features = sentence_features.unsqueeze(-1) * mask.unsqueeze(-1).expand_as(sentence_features.unsqueeze(-1)).float()
        # shape = [global_batch_size, global_batch_size]
        return sentence_features.unsqueeze(-1).float()
    

    
        
    def forward_null_text(self, data, device=None):
        
        device = self.device

        
        descriptions_text = []
        for index, frags in enumerate(data):
            des = ""
            descriptions_text.append(des)
            
            # continue
        
        text = clip.tokenize(descriptions_text,context_length=self.context_length).to(device)
        # 
      
        text_features = self.encode_text(text)
        
        
        sentence_features = self.transformer_fusion(text_features)
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        # shape = [global_batch_size, global_batch_size]
        
        return sentence_features.unsqueeze(-1).float()
    
    

class DTF(nn.Module):
    def __init__(self, 
                 input_dim,
                 fc_1_dim,
                 fc_2_dim,
                 fc_3_dim,
                 dropout,
                 transformer_dropout,
                 fusion_mode,
                 channels=128,
                 r=4):
        super(DTF, self).__init__()
        inter_channels = int(channels // r)

        self.att1 = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels)
        )

        self.att2 = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels)
        )

        self.sigmoid = nn.Sigmoid()

        self.fusion_mode = fusion_mode

        if fusion_mode == "concat":
            # input_dim = input_dim[0]+input_dim[1]
            input_dim = input_dim[0]

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

    def forward(self, fd, fp):
        # pdb.set_trace()
        w1 = self.sigmoid(self.att1(fd + fp))
        fout1 = fd * w1 + fp * (1 - w1)

        w2 = self.sigmoid(self.att2(fout1))
        fout2 = fd * w2 + fp * (1 - w2)

        x = torch.unsqueeze(fout2, 1)
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


        # return fout2

class TransDTA(nn.Module):
    def __init__(self, config, config_model_x, config_model_adj, device, train_flags, target_list, PRECISION=1, embed_dim=128):
        super().__init__()
        clip.available_models()

        self.config = config
        self.config_model_x = config_model_x
        self.config_model_adj = config_model_adj
        self.device = device
        self.target_list = target_list

        self.train_flags = train_flags
        
        # self.drug_module
        self.init_drug_module()

        # self.fusion_module
        self.init_fusion_module(self.config['model'])

        filter_num = 32
        out_dim = 1
        self.protein_encoder = TargetRepresentation()

        self.device = device
        self.PRECISION = PRECISION

        #=========================================================
        vocab_size = 50000
        transformer_width = 256
        self.context_num_length = 100
        self.context_length = 300
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
        self.drug_fusion_fc = torch.nn.Linear(256, 128)
        self.relu = torch.nn.ReLU()
        
        self.transformer_fusion = MLP(embed_dim*2, 512, embed_dim*2)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

        self.VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }


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

    def init_drug_module(self):
        max_feat_num, depth, nhid = self.config_model_x['max_feat_num'], self.config_model_x['depth'], self.config_model_x['nhid']
        # max_feat_num, depth, nhid = 10, 2, 16
        self.drug_x_module = ScoreNetworkX(max_feat_num, depth, nhid)
        
        max_feat_num, max_node_num, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads, conv =  self.config_model_adj['max_feat_num'], self.config_model_adj['max_node_num'], self.config_model_adj['nhid'], self.config_model_adj['num_layers'], self.config_model_adj['num_linears'], self.config_model_adj['c_init'], self.config_model_adj['c_hid'], self.config_model_adj['c_final'], self.config_model_adj['adim'], self.config_model_adj['num_heads'], self.config_model_adj['conv']
        # max_feat_num, max_node_num, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads, conv = 10, 100
        self.drug_adj_module = ScoreNetworkA(max_feat_num, max_node_num, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads, conv)

    def init_fusion_module(self, config):
        input_dim = [config['drug_module']['fc_2_dim'],
                     config['target_module']['fc_1_dim']]
        fc_1_dim = config['fusion_module']['fc_1_dim']
        fc_2_dim = config['fusion_module']['fc_2_dim']
        fc_3_dim = config['fusion_module']['fc_3_dim']
        dropout = config['fusion_module']['dropout']
        transformer_dropout = config['fusion_module']['transformer_dropout']
        fusion_mode = config['fusion_module']['fusion_mode']

        self.fusion_module = DTF(input_dim,
                                    fc_1_dim,
                                    fc_2_dim,
                                    fc_3_dim,
                                    dropout,
                                    transformer_dropout,
                                    fusion_mode)

    def forward4gen(self, x, adj, target_name, kd):
            
        device = x.device
        # pdb.set_trace()
        with torch.no_grad():
            pdb.set_trace()
            x_drug = self.drug_x_module(x, adj, None)
            adj_drug = self.drug_adj_module(x, adj, None)
            
        # target = [torch.FloatTensor(self.cell_matrix[1][self.cell_matrix[0][str(int(i))]]).reshape(1,1,735) for i in target_name]
        # target = torch.cat(target,dim=0).to(device)
        target = [torch.IntTensor(self.target_feature[int(i)]).reshape(1,1200) for i in target_name]
        target = torch.cat(target,dim=0).to(device)
        # pdb.set_trace()
        
        # cell_feature = self.target_module(target)
        
        x_drug = x_drug.reshape(-1, x_drug.shape[1] * x_drug.shape[2])
        x_drug = self.relu(self.drug_x_fc(x_drug))
        
        adj_drug = self.relu(self.drug_adj_fc1(adj_drug))
        adj_drug = adj_drug.reshape(-1, adj_drug.shape[1] * adj_drug.shape[2])
        adj_drug = self.relu(self.drug_adj_fc2(adj_drug))
        
        drug_feature = self.relu(self.drug_fusion_fc(torch.cat([x_drug, adj_drug], dim=1)))
        # pdb.set_trace()
        # target = self.target_list[]
        target_feature = self.protein_encoder(target)
        
     
        _, fusion_features, fusion_number_features = self.fusion_module(drug_feature, target_feature)
        
        # pdb.set_trace()
        descriptions_text = []
        descriptions_number = []
        for index, item in enumerate(kd):
            # pdb.set_trace()
            
            # des =  str(translateNumberToEnglish(item.item()))
            
            # des = ""
            # temp = int(item)
            # alpha = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            # while temp > 0:
            #     des = alpha[temp%10] + " " + des
            #     temp //= 10
            # if des == "":
            #     des += "zero "
            # des = des + "point" + num2english(item.item(), PRECISION=self.PRECISION)
            des = "zero point" + num2english(item.item(), PRECISION=self.PRECISION)
            descriptions_number.append(des)
            
            des = "The drug response value with protein No." + str(target_name[index].int().item()) +" is "
            descriptions_text.append(des)
            # pdb.set_trace()
            # continue
        
        text = clip.tokenize(descriptions_text,context_length=self.context_length).to(device)
        number = clip.tokenize(descriptions_number,context_length=self.context_num_length).to(device)
        # 
      
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
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
        return  logits_per_dc, logits_per_text, num_logits_per_dc, num_logits_per_text


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

    def get_target_feature(self, path):
        # def seqs2int(self, target):
        #     return [self.VOCAB_PROTEIN[s] for s in target.upper()]
        proteins = json.load(open(path), object_pairs_hook=OrderedDict)
        target_feature_list = []
        target_len = 1200
        for sequence in proteins.values():
            target = [self.VOCAB_PROTEIN[s] for s in sequence.upper()]
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]
            target_feature_list.append(target)
        self.target_feature = target_feature_list
    
    def generate_samples(self, start, end, dataset, PRECISION=1):
        # data: list, consist of [drug smile, cell line, ic50]
        descriptions_number = []
        # assert end - start == 1000
        
        # if model.training:    
        for ic50 in range(start,end,1):
        # for idx, ic50 in enumerate(range(0,1000,1)):
            des = "zero point " + num2english(ic50/(1*(PRECISION**2)), PRECISION)
            descriptions_number.append(des)

        text = clip.tokenize(descriptions_number, context_length = 100).to(self.device)
        # pdb.set_trace()
        text_features = self.encode_num(text)
        
        return text_features


    def text_forward(self, data, device=None):

        # target_name, kd
        device = self.device
        
       
        # pdb.set_trace()
        descriptions_text = []
        descriptions_number = []

        for index, item in enumerate(data):
            
            
            # des =  str(translateNumberToEnglish(item.item()))
            des = "zero point" + num2english(item[0][1].item(), PRECISION=self.PRECISION)
            descriptions_number.append(des)
            
            des = "The drug response value with protein No." + str(item[0][0].int().item()) +" is "
            descriptions_text.append(des)
            
            
            # continue
        
        text = clip.tokenize(descriptions_text,context_length=self.context_length).to(device)
        number = clip.tokenize(descriptions_number,context_length=self.context_num_length).to(device)
        # 
      
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        # pdb.set_trace()
        mask = self.create_mask(sentence_features.unsqueeze(-1),0.8)
        
        sentence_features = sentence_features.unsqueeze(-1) * mask.unsqueeze(-1).expand_as(sentence_features.unsqueeze(-1)).float()
        # shape = [global_batch_size, global_batch_size]
        return sentence_features


    
    def multi_data_text_forward(self, data, device=None):

        # target_name, kd
        device = self.device
        
       
        # pdb.set_trace()
        descriptions_text = []
        descriptions_number = []

        for index, item in enumerate(data):
            
            if item[0][0] == 0 and item[0][1] == 0:
                descriptions_number.append("")
                descriptions_text.append("")
            else:
                # des =  str(translateNumberToEnglish(item.item()))
                des = "zero point" + num2english(item[0][1].item(), PRECISION=self.PRECISION)
                descriptions_number.append(des)
                
                des = "The drug response value with protein No." + str(item[0][0].int().item()) +" is "
                descriptions_text.append(des)
                
                
            # continue
        
        text = clip.tokenize(descriptions_text,context_length=self.context_length).to(device)
        number = clip.tokenize(descriptions_number,context_length=self.context_num_length).to(device)
        # 
      
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        return sentence_features.unsqueeze(-1).float()


    def create_mask(self, embeddings, word_dropout):
        # word_dropout =0 then mask is not useful, mask will be ones matrix. 
        batch, length, size = embeddings.size()
        mask = embeddings.new_empty(batch, 1)
        mask = mask.bernoulli_(1 - word_dropout)
        # embeddings = embeddings * mask.unsqueeze(-1).expand_as(embeddings).float()
        return mask

    
    def text_full_forward(self, data, device=None):
        
        
        
        device = self.device
      
        descriptions_text = []
        descriptions_number = []
        for index, item in enumerate(data):

            des = "zero point" + num2english(item[0][1].item(), PRECISION=self.PRECISION)
            descriptions_number.append(des)
            
            des = "The drug response value with protein No." + str(item[0][0].int().item()) +" is "
            descriptions_text.append(des)
                
            
            
        text = clip.tokenize(descriptions_text,context_length=self.context_length).to(device)
        number = clip.tokenize(descriptions_number,context_length=self.context_num_length).to(device)
        # 
   
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        return sentence_features.unsqueeze(-1).float()
    
    
    def forward_null_text(self, data, device=None):
        device = self.device
       
        descriptions_text = []
        descriptions_number = []
        for index, item in enumerate(data):

            des = ""
            descriptions_number.append(des)
   
            des = ""
            descriptions_text.append(des)
            
        text = clip.tokenize(descriptions_text,context_length=self.context_length).to(device)
        number = clip.tokenize(descriptions_number,context_length=self.context_num_length).to(device)
        # 
      
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        return sentence_features.unsqueeze(-1).float()




class Conv1dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):

        return self.inc(x)


class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):

        return self.inc(x).squeeze(-1)


class TargetRepresentation(nn.Module):
    def __init__(self, block_num=3, vocab_size=25+1, embedding_num=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(1, block_num + 1):
            self.block_list.append(
                StackCNN(block_idx, embedding_num, 128, 3)
            )
        self.linear = nn.Linear(block_num * 128, 128)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9)
        self.bn3 = nn.BatchNorm1d(128)
        self.advpool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, v):
        # pdb.set_trace()
        v = self.embed(v)
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        # pdb.set_trace()
        v = self.advpool(v).squeeze(-1)
        
        return v



def csv_to_dict(filename):
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row if it exists
        for row in csvreader:
            entry = {'Interaction type': row[0], 'Description': row[1]}
            data.append(entry)
    return data

def find_description(entries, interaction_type):
    for entry in entries:
        if entry['Interaction type'] == interaction_type:
            return entry['Description']
    return None  # Return None if interaction type not found



class TransDDI(torch.nn.Module):
    def __init__(self, config, config_model_x, config_model_adj, device, train_flags):
        super(TransDDI, self).__init__()
        clip.available_models()

        self.config = config
        self.config_model_x = config_model_x
        self.config_model_adj = config_model_adj
        self.device = device

        self.train_flags = train_flags
        # pdb.set_trace()
        self.effect_id_to_string = csv_to_dict(config['interaction'])
        
        # self.drug_module
        self.init_drug_module()
       
        vocab_size = 50000
        transformer_width = 256
        # self.context_length = context_length
        self.context_length = 1200

        transformer_width = 128
        transformer_layers = 3
        transformer_heads = 4
        embed_dim = 128
        
        # text encode
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
                   
        self.drug_dim = 100
        self.drug_x_fc = torch.nn.Linear(self.drug_dim*34, 128)
        self.drug_adj_fc1 = torch.nn.Linear(self.drug_dim, 16)
        self.drug_adj_fc2 = torch.nn.Linear(self.drug_dim*16, 128)
        self.drug_fusion_fc = torch.nn.Linear(256, 128)
        self.relu = torch.nn.ReLU()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        
        self.transformer_fusion = MLP(embed_dim, 256, embed_dim)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
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
                       
    def init_drug_module(self):
        max_feat_num, depth, nhid = self.config_model_x['max_feat_num'], self.config_model_x['depth'], self.config_model_x['nhid']
        self.drug_x_module = ScoreNetworkX(max_feat_num, depth, nhid)
        
        max_feat_num, max_node_num, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads, conv =  self.config_model_adj['max_feat_num'], self.config_model_adj['max_node_num'], self.config_model_adj['nhid'], self.config_model_adj['num_layers'], self.config_model_adj['num_linears'], self.config_model_adj['c_init'], self.config_model_adj['c_hid'], self.config_model_adj['c_final'], self.config_model_adj['adim'], self.config_model_adj['num_heads'], self.config_model_adj['conv']
        self.drug_adj_module = ScoreNetworkA(max_feat_num, max_node_num, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads, conv)

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
 
    def forward4gen(self, x, adj, drug2, effect):
            
        device = x.device
        # pdb.set_trace()
        with torch.no_grad():
            x_drug = self.drug_x_module(x, adj, None)
            adj_drug = self.drug_adj_module(x, adj, None)
      
        x_drug = x_drug.reshape(-1, x_drug.shape[1] * x_drug.shape[2])
        x_drug = self.relu(self.drug_x_fc(x_drug))
        
        # pdb.set_trace()
        adj_drug = self.relu(self.drug_adj_fc1(adj_drug))
        adj_drug = adj_drug.reshape(-1, adj_drug.shape[1] * adj_drug.shape[2])
        adj_drug = self.relu(self.drug_adj_fc2(adj_drug))
        
        drug_feature = self.relu(self.drug_fusion_fc(torch.cat([x_drug, adj_drug], dim=1)))
        
        # pdb.set_trace()
        descriptions_text = []
        
        for drug2_, effect_ in zip(drug2, effect):
          
            des = find_description(self.effect_id_to_string, str(effect_.int().item()))
            replacement_dict = {"#Drug1": "this drug", "#Drug2": drug2_}

            for old_string, new_string in replacement_dict.items():
                des = des.replace(old_string, new_string)
            descriptions_text.append(des)
            
            # continue
        
        text = clip.tokenize(descriptions_text,context_length=self.context_length).to(device)
      
        text_features = self.encode_text(text)
        
        sentence_features = self.transformer_fusion(text_features)
        # normalized features
        drug_feature = drug_feature / drug_feature.norm(dim=1, keepdim=True)
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        # pdb.set_trace()
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_df = logit_scale * drug_feature @ sentence_features.t()
        logits_per_text = logits_per_df.t()
        
        # shape = [global_batch_size, global_batch_size]
        return  logits_per_df, logits_per_text
                 
    def text_forward(self, data, device=None):
        
        device = self.device
      
        descriptions_text = []
        descriptions_number = []
        for index, item in enumerate(data):
            # des =  str(translateNumberToEnglish(item.item()))
            # import pdb;pdb.set_trace()
            des = "zero point" + num2english(item[0][1].item())
            descriptions_number.append(des)
            
            des = "The drug response value about " + str(item[0][0].int().item()) +" is "
            descriptions_text.append(des)
            
        text = clip.tokenize(descriptions_text,context_length=300).to(device)
        number = clip.tokenize(descriptions_number,context_length=100).to(device)
        # 
      
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        mask = self.create_mask(sentence_features.unsqueeze(-1),0.5)
        
        sentence_features = sentence_features.unsqueeze(-1) * mask.unsqueeze(-1).expand_as(sentence_features.unsqueeze(-1)).float()
        # shape = [global_batch_size, global_batch_size]
        return sentence_features
    
    
    def text_full_forward(self, data, device=None):
        
        device = self.device
      
        descriptions_text = []

        for index, frags in enumerate(data):
            # frags = onehot_to_string(item.cpu().numpy())
            if '?' in frags:
                frags = frags.replace('?', ', ')
            # pdb.set_trace()
            des = 'This drug contains ' + frags
            descriptions_text.append(des)
            
        text = clip.tokenize(descriptions_text,context_length=self.context_length).to(device)
        # 
      
        text_features = self.encode_text(text)
        
        sentence_features = self.transformer_fusion(text_features)
        # normalized features
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)


        # sentence_features = sentence_features.unsqueeze(-1) * mask.unsqueeze(-1).expand_as(sentence_features.unsqueeze(-1)).float()
        # shape = [global_batch_size, global_batch_size]
        return sentence_features.unsqueeze(-1).float()
    
