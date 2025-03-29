import torch
from torch.nn import Parameter
import torch.nn.functional as F

from models.layers import DenseGCNConv, MLP
from utils.graph_utils import mask_adjs, pow_tensor
from models.attention import  AttentionLayer
from models.control_utils import *


class BaselineNetworkLayer(torch.nn.Module):

    def __init__(self, num_linears, conv_input_dim, conv_output_dim, input_dim, output_dim, batch_norm=False):

        super(BaselineNetworkLayer, self).__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.convs.append(DenseGCNConv(conv_input_dim, conv_output_dim))
        self.hidden_dim = max(input_dim, output_dim)
        self.mlp_in_dim = input_dim + 2*conv_output_dim
        self.mlp = MLP(num_linears, self.mlp_in_dim, self.hidden_dim, output_dim, 
                            use_bn=False, activate_func=F.elu)
        self.multi_channel = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim, 
                                    use_bn=False, activate_func=F.elu)
        
    def forward(self, x, adj, flags):
    
        x_list = []
        for _ in range(len(self.convs)):
            _x = self.convs[_](x, adj[:,_,:,:])
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)) , flags)
        x_out = torch.tanh(x_out)

        x_matrix = node_feature_to_matrix(x_out)
        mlp_in = torch.cat([x_matrix, adj.permute(0,2,3,1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0,3,1,2)
        _adj = _adj + _adj.transpose(-1,-2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out


class BaselineNetwork(torch.nn.Module):

    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                    c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

        super(BaselineNetwork, self).__init__()

        self.nfeat = max_feat_num
        self.max_node_num = max_node_num
        self.nhid  = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _==0:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nfeat, self.nhid, self.c_init, self.c_hid))

            elif _==self.num_layers-1:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_final))

            else:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_hid)) 

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
                            use_bn=False, activate_func=F.elu)
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)   

    def forward(self, x, adj, flags=None):

        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):

            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)
        
        adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        out_shape = adjs.shape[:-1] # B x N x N
        score = self.final(adjs).view(*out_shape)

        self.mask = self.mask.to(score.device)
        score = score * self.mask

        score = mask_adjs(score, flags)

        return score


class ScoreNetworkA_control(BaselineNetwork):

    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                    c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

        super(ScoreNetworkA_control, self).__init__(max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                                            c_init, c_hid, c_final, adim, num_heads=4, conv='GCN')
        
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.dims = 1
        
        self.layers = torch.nn.ModuleList()
        self.control_layers = torch.nn.ModuleList()
        
        self.zero_convs_gnn_x = torch.nn.ModuleList()
        self.zero_convs_gnn_adj = torch.nn.ModuleList()
        
        self.zero_convs_gnn_x.append(self.make_zero_conv(100))
        self.zero_convs_gnn_adj.append(self.make_zero_conv(100,input_type="adj"))
        
        for _ in range(self.num_layers):
            self.zero_convs_gnn_x.append(self.make_zero_conv(100))
            self.zero_convs_gnn_adj.append(self.make_zero_conv(100,input_type="adj"))
            
            if _==0:
                self.layers.append(AttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init, 
                                                    self.c_hid, self.num_heads, self.conv))
                self.control_layers.append(AttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init, 
                                                    self.c_hid, self.num_heads, self.conv))
            elif _==self.num_layers-1:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_final, self.num_heads, self.conv))
                self.control_layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_final, self.num_heads, self.conv))
            else:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_hid, self.num_heads, self.conv))
                self.control_layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_hid, self.num_heads, self.conv))

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
                            use_bn=False, activate_func=F.elu)
        # self.control_final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
        #                     use_bn=False, activate_func=F.elu)
        
        
        self.condition_embed_0 = torch.nn.Conv2d(in_channels=128,out_channels=100,kernel_size=1)
        self.condition_embed_1_x = linear(2, 10)
        self.condition_embed_1_adj = linear(2, 100)
        # self.condition_embed_2 = linear(1, self.fdim)
        # self.condition_embed_3 = linear(self.fdim, 1)
        
        # self.zero_convs_mlp = torch.nn.ModuleList()
        # self.zero_convs_mlp.append(self.make_zero_conv(100))
        # self.zero_convs_mlp.append(self.make_zero_conv(100))
        
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)  
        

        
    def make_zero_conv(self, channels, input_type=None):
        if input_type=='adj':
            return TimestepEmbedSequential(zero_module(conv_nd(self.dims+1, channels, channels, 1, padding=0)))
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, adj, flags):
        # print("121:{}".format(torch.cuda.max_memory_reserved(0)))
        adjc = pow_tensor(adj, self.c_init)
        # print("122:{}".format(torch.cuda.max_memory_reserved(0)))
        adj_list = [adjc]
        for _ in range(self.num_layers):
            x, adjc = self.layers[_](x, adjc, flags)
            # print("23—:{}".format(torch.cuda.max_memory_reserved(0)))
            adj_list.append(adjc)
            # print("23—:{}".format(torch.cuda.max_memory_reserved(0)))
        
        # print("124:{}".format(torch.cuda.max_memory_reserved(0)))
        adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        del adj_list
        torch.cuda.empty_cache()
        
        # print("125:{}".format(torch.cuda.max_memory_reserved(0)))
        out_shape = adjs.shape[:-1] # B x N x N
        score = self.final(adjs).view(*out_shape)
        self.mask = self.mask.to(score.device)
        score = score * self.mask
        score = mask_adjs(score, flags)
        return score

    # def forward_c(self, x, adj, cond, flags, t):
        
        
        
    #     adjc = pow_tensor(adj, self.c_init)
    #     adjc_clone = adjc.clone()
    #     x_clone = x.clone()
    #     adj_list_raw = [adjc_clone]
        
    #     with torch.no_grad():
    #         for _ in range(self.num_layers):
    #             x_clone, adjc_clone = self.layers[_](x_clone, adjc_clone, flags)
    #             adj_list_raw.append(adjc_clone)

    #         adjc_clone = torch.cat(adj_list_raw, dim=1).permute(0,2,3,1)
    #         del adj_list_raw
    #         torch.cuda.empty_cache()
            
    #         out_shape = adjc_clone.shape[:-1] # B x N x N
    #         score_clone = self.final(adjc_clone).view(*out_shape)
    #         self.mask = self.mask.to(score_clone.device)
    #         score_clone = score_clone * self.mask
    #         score_clone = mask_adjs(score_clone, flags)
        
        
    #     # control
    #     cond = cond.reshape(x.shape[0],128,-1).unsqueeze(3)
    #     cond = self.condition_embed_0(cond).squeeze(3)
    #     cond = self.condition_embed_1(cond)
    #     adj_list = [adjc]
        
    #     for module, zero_conv in zip(self.control_layers, self.zero_convs_gnn):
    #         cond = zero_conv(cond)
    #         adjc = adjc+cond.unsqueeze(1)
    #         x, adjc = module(x, adjc, flags)
    #         adj_list.append(adjc)


    #     adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
    #     del adj_list
    #     torch.cuda.empty_cache()
        
    #     out_shape = adjs.shape[:-1] # B x N x N
        
        
    #     cond = self.zero_convs_mlp[0](cond)
    #     cond = self.condition_embed_2(cond.unsqueeze(3))
        
    #     adjs = adjs+cond
        
    #     score = self.control_final(adjs).view(*out_shape)
        
                
    #     cond = self.condition_embed_3(cond).squeeze(3)
    #     cond = self.zero_convs_mlp[1](cond)
        
    #     score = score+cond
        
    #     self.mask = self.mask.to(score.device)
    #     score = score * self.mask
    #     score = mask_adjs(score, flags)
        
    #     # import pdb;pdb.set_trace()
    #     # Guidance Scale
    #     score = score_clone + self.w_cfg * (score-score_clone)
    #     return score
    
    
    def forward_c(self, x, adj, cond, flags, t):
        
        # if cond.shape[1] ==128:
            
        #     cond = cond.reshape(x.shape[0], 128, -1)
        #     # drug-tag, without x. Then drug dim is 128, text/condition dim is 128.
        #     cond = torch.cat((cond,cond),dim=2)

        cond = cond.reshape(x.shape[0], 128, -1).unsqueeze(3)
        cond = self.condition_embed_0(cond).squeeze(3)
        
        
        cond_x = self.condition_embed_1_x(cond)
        cond_adj = self.condition_embed_1_adj(cond) 


        adjc = pow_tensor(adj, self.c_init)
        cond_adj = pow_tensor(cond_adj, self.c_init)
        adj_list = [adjc]
        
        cond_x = self.zero_convs_gnn_x[0](cond_x)
        cond_adj = self.zero_convs_gnn_adj[0](cond_adj.permute(0,2,3,1)).permute(0,3,1,2)
   
        
        cond_x = x + cond_x
        cond_adj = adjc + cond_adj
        
        
        for _ in range(self.num_layers):
    
            x, adjc = self.layers[_](x, adjc, flags)
            cond_x, cond_adj = self.control_layers[_](cond_x, cond_adj, flags)
            
            cond_x = self.zero_convs_gnn_x[_+1](cond_x)
            cond_adj = self.zero_convs_gnn_adj[_+1](cond_adj.permute(0,2,3,1)).permute(0,3,1,2)
            
            x = x + cond_x
            adjc = adjc + cond_adj
            
            adj_list.append(adjc)

        adjc = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        # del adj_list
        # torch.cuda.empty_cache()
        
        out_shape = adjc.shape[:-1] # B x N x N

        score = self.final(adjc).view(*out_shape)
        self.mask = self.mask.to(score.device)
        score = score * self.mask
        score = mask_adjs(score, flags)
        
    
        return score
