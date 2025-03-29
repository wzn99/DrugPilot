import torch
import torch.nn.functional as F

from models.layers import DenseGCNConv, MLP
from utils.graph_utils import mask_x, pow_tensor
from models.attention import  AttentionLayer
from models.control_utils import *
 
class ScoreNetworkX_control(torch.nn.Module):

    def __init__(self, max_feat_num, depth, nhid):

        super(ScoreNetworkX_control, self).__init__()

        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid
        self.dims = 1
        
        self.layers = torch.nn.ModuleList()
        self.control_layers = torch.nn.ModuleList()
        self.zero_convs_gnn = torch.nn.ModuleList()
        
        self.zero_convs_gnn.append(self.make_zero_conv(100))
        
        for _ in range(self.depth):
            self.zero_convs_gnn.append(self.make_zero_conv(100))
            
            if _ == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
                self.control_layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))
                self.control_layers.append(DenseGCNConv(self.nhid, self.nhid))
              

        self.fdim = self.nfeat + self.depth * (self.nhid) 
        
        
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat, 
                            use_bn=False, activate_func=F.elu)
        
        self.condition_embed_0 = torch.nn.Conv2d(in_channels=128, out_channels=100, kernel_size=1, stride =(1,1))
        self.condition_embed_1 = linear(2, self.nfeat)
        self.condition_embed_1_t = linear(1, 2*100)
        # self.condition_embed_1_1 = linear(self.nfeat, self.nhid)
        # self.condition_embed_2 = linear(self.nhid, self.fdim)
        # self.condition_embed_3 = linear(self.fdim, self.nfeat)
        
        # self.zero_convs_mlp = torch.nn.ModuleList()
        # self.zero_convs_mlp.append(self.make_zero_conv(100))
        # self.zero_convs_mlp.append(self.make_zero_conv(100))
        
        self.activation = torch.tanh

        # self.model_channels = 100
        
        # self.time_embed_dim = self.model_channels * 4
        # self.time_embed = nn.Sequential(
        #     linear(self.model_channels, self.time_embed_dim),
        #     nn.SiLU(),
        #     linear(self.time_embed_dim, self.time_embed_dim),
        # )
        

        
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, adj, flags):
        
        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)

        x = mask_x(x, flags)

        return x

    # def forward_c(self, x, adj, cond, flags, t):

    #     x_clone = x.clone()
    #     x_list_raw = [x_clone]
        
    #     with torch.no_grad():
    #         for _ in range(self.depth):
    #             x_clone = self.layers[_](x_clone, adj)
    #             x_clone = self.activation(x_clone)
    #             x_list_raw.append(x_clone)

    #         xs_raw = torch.cat(x_list_raw, dim=-1) # B x N x (F + num_layers x H)
    #         del x_list_raw
    #         torch.cuda.empty_cache()
            
    #         out_shape = (adj.shape[0], adj.shape[1], -1)
    #         x_clone = self.final(xs_raw).view(*out_shape)
    #         x_clone = mask_x(x_clone, flags)

    #     # control
        
    #     cond = cond.reshape(x.shape[0], 128, -1).unsqueeze(3)
    #     cond = self.condition_embed_0(cond).squeeze(3)
    #     cond = self.condition_embed_1(cond)
    #     x_list = [x]

    #     for module, zero_conv in zip(self.control_layers, self.zero_convs_gnn):
    #         if x.shape[-1] == self.nhid:
    #             cond = self.condition_embed_1_1(cond)
                
    #         cond = zero_conv(cond)
    #         x = x+cond
    #         x = module(x, adj)
    #         x_list.append(x)
            
    #     xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
    #     del x_list
    #     torch.cuda.empty_cache()
        
    #     cond = self.zero_convs_mlp[0](cond)
    #     cond = self.condition_embed_2(cond)
        
    #     xs = xs+cond
    #     x = self.control_final(xs).view(*out_shape)
        
    #     cond = self.condition_embed_3(cond)
    #     cond = self.zero_convs_mlp[1](cond)
        
    #     x = x+cond
    #     x = mask_x(x, flags)
        
        
    #     # Guidance Scale
    #     x = x_clone + self.w_cfg * (x-x_clone)
        
    #     return x
    
    
    def forward_c(self, x, adj, cond, flags, t):


        cond = cond.reshape(x.shape[0], 128, -1).unsqueeze(3)
        cond = self.condition_embed_0(cond).squeeze(3)
        cond = self.activation(cond)
        
        t = self.condition_embed_1_t(t.reshape(x.shape[0],1,1)).reshape(x.shape[0],100,2)
        cond = cond + t
        cond = self.activation(cond)
        
        cond = self.condition_embed_1(cond)
        cond = self.zero_convs_gnn[0](cond)
        cond = x+cond
        x_list = [x]
        
        for _ in range(self.depth):
          
            x = self.layers[_](x, adj)
            x = self.activation(x)
            
            cond = self.control_layers[_](cond, adj)
            cond = self.activation(cond)
            cond = self.zero_convs_gnn[_+1](cond)
            x = x+cond
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        # del x_list
        # torch.cuda.empty_cache()
        
        out_shape = (adj.shape[0], adj.shape[1], -1)
    
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)
        
        return x

class ScoreNetworkX_GMH(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, num_linears,
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
        super().__init__()

        self.depth = depth
        self.c_init = c_init

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(AttentionLayer(num_linears, max_feat_num, nhid, nhid, c_init, 
                                                  c_hid, num_heads, conv))
            elif _ == self.depth - 1:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_final, num_heads, conv))
            else:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_hid, num_heads, conv))

        fdim = max_feat_num + depth * nhid
        self.final = MLP(num_layers=3, input_dim=fdim, hidden_dim=2*fdim, output_dim=max_feat_num, 
                         use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags):
        adjc = pow_tensor(adj, self.c_init)

        x_list = [x]
        for _ in range(self.depth):
            x, adjc = self.layers[_](x, adjc, flags)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)

        return x
