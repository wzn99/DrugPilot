import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, global_mean_pool as gep, global_sort_pool
from torch_geometric.utils import dropout_adj, softmax

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(Attention, self).__init__()
        self.project_x = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_xt = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_xq = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, x, xt, xq):
        x = self.project_x(x)
        xt = self.project_xt(xt)
        xq = self.project_xq(xq)
        a = torch.cat((x, xt, xq), 1)
        a = torch.softmax(a, dim=1)
        return a

class MSGNN_DTA(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, num_features_xc=92, embed_dim=128, output_dim=128, n_filters=32, dropout=0.2):
        super(MSGNN_DTA, self).__init__()

        print('DTA_GAT Loading ...')
        self.n_output = n_output
        self.mol_conv = nn.ModuleList([])
        self.mol_conv.append(GATConv(num_features_mol, num_features_mol *4, heads=2, dropout=dropout, concat=False))
        self.mol_conv.append(GATConv(num_features_mol* 4, num_features_mol* 4, heads=2, dropout=dropout, concat=False))
        self.mol_conv.append(GATConv(num_features_mol* 4, num_features_mol* 4, heads=2, dropout=dropout, concat=False))
        self.mol_out_feats = num_features_mol * 4
        self.mol_seq_fc1 = nn.Linear(num_features_mol*4, num_features_mol*4)
        self.mol_seq_fc2 = nn.Linear(num_features_mol*4, num_features_mol*4)
        self.mol_bias = nn.Parameter(torch.rand(1, num_features_mol*4))
        torch.nn.init.uniform_(self.mol_bias, a=-0.2, b=0.2)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv = nn.ModuleList([])
        self.pro_conv.append(GCNConv(num_features_pro, num_features_pro * 4))
        self.pro_conv.append(GATConv(num_features_pro * 4, num_features_pro * 4, heads=2, dropout=dropout, concat=False))
        self.pro_conv.append(GATConv(num_features_pro * 4, num_features_pro * 4, heads=2, dropout=dropout, concat=False))
        self.pro_out_feats = num_features_pro * 4
        self.pro_seq_fc1 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        self.pro_seq_fc2 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        self.pro_bias = nn.Parameter(torch.rand(1, num_features_pro * 4))
        torch.nn.init.uniform_(self.pro_bias, a=-0.2, b=0.2)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.clique_conv = nn.ModuleList([])
        self.clique_conv.append(GATConv(num_features_xc, num_features_xc * 4, heads=2, dropout=dropout, concat=False))
        self.clique_conv.append(GATConv(num_features_xc * 4, num_features_xc * 4, heads=2, dropout=dropout, concat=False))
        self.clique_conv.append(GATConv(num_features_xc * 4, num_features_xc * 4, heads=2, dropout=dropout, concat=False))
        self.clique_out_feats = num_features_xc * 4
        self.clique_seq_fc1 = nn.Linear(num_features_xc * 4, num_features_xc * 4)
        self.clique_seq_fc2 = nn.Linear(num_features_xc * 4, num_features_xc * 4)
        self.clique_bias = nn.Parameter(torch.rand(1, num_features_xc * 4))
        torch.nn.init.uniform_(self.clique_bias, a=-0.2, b=0.2)
        self.clique_fc_g1 = torch.nn.Linear(num_features_xc * 4, 1024)
        self.clique_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.attention = Attention(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(3 * output_dim, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.out = nn.Linear(512, self.n_output)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data_mol, data_pro, data_clique):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        # get clique input
        clique_x, clique_edge_index, clique_batch = data_clique.x, data_clique.edge_index,data_clique.batch

        mol_n = mol_x.size(0)
        for i in range(len(self.mol_conv)):
            x = self.mol_conv[i](mol_x, mol_edge_index)
            if i < len(self.mol_conv)-1:
                x = self.relu(x)
            if i==0:
                mol_x = x
                continue
            mol_z = torch.sigmoid(self.mol_seq_fc1(x) + self.mol_seq_fc2(mol_x) + self.mol_bias.expand(mol_n, self.mol_out_feats))
            mol_x = mol_z * x + (1 - mol_z) * mol_x

        x = gep(mol_x, mol_batch)  # global pooling
        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        pro_n = target_x.size(0)
        for i in range(len(self.pro_conv)):
            if i == 0:
                xt = self.pro_conv[i](target_x, target_edge_index, target_weight)
            else:
                xt = self.pro_conv[i](target_x, target_edge_index)
            if i < len(self.pro_conv) - 1:
                xt = self.relu(xt)
            if i == 0:
                target_x = xt
                continue
            pro_z = torch.sigmoid(
                self.pro_seq_fc1(xt) + self.pro_seq_fc2(target_x) + self.pro_bias.expand(pro_n, self.pro_out_feats))
            target_x = pro_z * xt + (1 - pro_z) * target_x

        xt = gep(target_x, target_batch)  # global pooling
        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # clique graph embedding
        xq_n = clique_x.size(0)
        for i in range(len(self.clique_conv)):
            xq = self.clique_conv[i](clique_x, clique_edge_index)
            if i < len(self.clique_conv) - 1:
                xq = self.relu(xq)
            if i == 0:
                clique_x = xq
                continue
            clique_z = torch.sigmoid(
                self.clique_seq_fc1(xq) + self.clique_seq_fc2(clique_x) + self.clique_bias.expand(xq_n,
                                                                                                  self.clique_out_feats))
            clique_x = clique_z * xq + (1 - clique_z) * clique_x

        xq = gep(clique_x, clique_batch)  # global max pooling
        # flatten
        xq = self.relu(self.clique_fc_g1(xq))
        xq = self.dropout(xq)
        xq = self.clique_fc_g2(xq)
        xq = self.dropout(xq)
        # concat
        a = self.attention(x, xt, xq)
        emb = torch.stack([x, xt, xq], dim=1)
        a = a.unsqueeze(dim=2)
        emb = (a * emb).reshape(-1, 3 * 128)
        # add some dense layers
        xc = self.fc1(emb)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class Graph_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, num_features_xc=92, embed_dim=128, output_dim=128, n_filters=32, dropout=0.2):
        super(Graph_GCN, self).__init__()

        print('DTA_GAT Loading ...')
        self.n_output = n_output
        self.mol_conv = nn.ModuleList([])
        self.mol_conv.append(GCNConv(num_features_mol, num_features_mol *4))
        self.mol_conv.append(GCNConv(num_features_mol* 4, num_features_mol* 4))
        self.mol_conv.append(GCNConv(num_features_mol* 4, num_features_mol* 4))
        self.mol_out_feats = num_features_mol * 4
        self.mol_seq_fc1 = nn.Linear(num_features_mol*4, num_features_mol*4)
        self.mol_seq_fc2 = nn.Linear(num_features_mol*4, num_features_mol*4)
        self.mol_bias = nn.Parameter(torch.rand(1, num_features_mol*4))
        torch.nn.init.uniform_(self.mol_bias, a=-0.2, b=0.2)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv = nn.ModuleList([])
        self.pro_conv.append(GCNConv(num_features_pro, num_features_pro * 4))
        self.pro_conv.append(GCNConv(num_features_pro * 4, num_features_pro * 4))
        self.pro_conv.append(GCNConv(num_features_pro * 4, num_features_pro * 4))
        self.pro_out_feats = num_features_pro * 4
        self.pro_seq_fc1 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        self.pro_seq_fc2 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        self.pro_bias = nn.Parameter(torch.rand(1, num_features_pro * 4))
        torch.nn.init.uniform_(self.pro_bias, a=-0.2, b=0.2)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.clique_conv = nn.ModuleList([])
        self.clique_conv.append(GCNConv(num_features_xc, num_features_xc * 4))
        self.clique_conv.append(GCNConv(num_features_xc * 4, num_features_xc * 4))
        self.clique_conv.append(GCNConv(num_features_xc * 4, num_features_xc * 4))
        self.clique_out_feats = num_features_xc * 4
        self.clique_seq_fc1 = nn.Linear(num_features_xc * 4, num_features_xc * 4)
        self.clique_seq_fc2 = nn.Linear(num_features_xc * 4, num_features_xc * 4)
        self.clique_bias = nn.Parameter(torch.rand(1, num_features_xc * 4))
        torch.nn.init.uniform_(self.clique_bias, a=-0.2, b=0.2)
        self.clique_fc_g1 = torch.nn.Linear(num_features_xc * 4, 1024)
        self.clique_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.attention = Attention(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(3 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro, data_clique):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        # get clique input
        clique_x, clique_edge_index, clique_batch = data_clique.x, data_clique.edge_index,data_clique.batch

        mol_n = mol_x.size(0)
        for i in range(len(self.mol_conv)):
            x = self.mol_conv[i](mol_x, mol_edge_index)
            if i < len(self.mol_conv)-1:
                x = self.relu(x)
            if i==0:
                mol_x = x
                continue
            mol_z = torch.sigmoid(self.mol_seq_fc1(x) + self.mol_seq_fc2(mol_x) + self.mol_bias.expand(mol_n, self.mol_out_feats))
            mol_x = mol_z * x + (1 - mol_z) * mol_x
        x = gmp(mol_x, mol_batch)  # global pooling
        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        pro_n = target_x.size(0)
        for i in range(len(self.pro_conv)):
            xt = self.pro_conv[i](target_x, target_edge_index, target_weight)
            if i < len(self.pro_conv) - 1:
                xt = self.relu(xt)
            if i == 0:
                target_x = xt
                continue
            pro_z = torch.sigmoid(
                self.pro_seq_fc1(xt) + self.pro_seq_fc2(target_x) + self.pro_bias.expand(pro_n, self.pro_out_feats))
            target_x = pro_z * xt + (1 - pro_z) * target_x
        xt = gmp(target_x, target_batch)  # global pooling
        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # clique graph embedding
        xq_n = clique_x.size(0)
        for i in range(len(self.clique_conv)):
            xq = self.clique_conv[i](clique_x, clique_edge_index)
            if i < len(self.clique_conv) - 1:
                xq = self.relu(xq)
            if i == 0:
                clique_x = xq
                continue
            clique_z = torch.sigmoid(
                self.clique_seq_fc1(xq) + self.clique_seq_fc2(clique_x) + self.clique_bias.expand(xq_n, self.clique_out_feats))
            clique_x = clique_z * xq + (1 - clique_z) * clique_x
        xq = gmp(clique_x, clique_batch)  # global max pooling
        # flatten
        xq = self.relu(self.clique_fc_g1(xq))
        xq = self.dropout(xq)
        xq = self.clique_fc_g2(xq)
        xq = self.dropout(xq)
        # concat
        a = self.attention(x,xt,xq)
        emb = torch.stack([x,xt,xq], dim=1)
        a= a.unsqueeze(dim =2)
        emb = (a * emb).reshape(-1, 3* 128)
        # add some dense layers
        xc = self.fc1(emb)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

class Graph_noSub(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, num_features_xc=92, embed_dim=128,
                 output_dim=128, n_filters=32, dropout=0.2):
        super(Graph_noSub, self).__init__()

        print('Graph_GCN_attension Loading ...')
        self.n_output = n_output
        self.mol_conv = nn.ModuleList([])
        self.mol_conv.append(GATConv(num_features_mol, num_features_mol * 4, heads=2, dropout=dropout, concat=False))
        self.mol_conv.append(GATConv(num_features_mol * 4, num_features_mol * 4, heads=2, dropout=dropout, concat=False))
        self.mol_conv.append(GATConv(num_features_mol * 4, num_features_mol * 4, heads=2, dropout=dropout, concat=False))
        self.mol_out_feats = num_features_mol * 4
        self.mol_seq_fc1 = nn.Linear(num_features_mol * 4, num_features_mol * 4)
        self.mol_seq_fc2 = nn.Linear(num_features_mol * 4, num_features_mol * 4)
        self.mol_bias = nn.Parameter(torch.rand(1, num_features_mol * 4))
        torch.nn.init.uniform_(self.mol_bias, a=-0.2, b=0.2)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv = nn.ModuleList([])
        self.pro_conv.append(GCNConv(num_features_pro, num_features_pro * 4))
        self.pro_conv.append(GATConv(num_features_pro * 4, num_features_pro * 4, heads=2, dropout=dropout, concat=False))
        self.pro_conv.append(GATConv(num_features_pro * 4, num_features_pro * 4, heads=2, dropout=dropout, concat=False))
        self.pro_out_feats = num_features_pro * 4
        self.pro_seq_fc1 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        self.pro_seq_fc2 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        self.pro_bias = nn.Parameter(torch.rand(1, num_features_pro * 4))
        torch.nn.init.uniform_(self.pro_bias, a=-0.2, b=0.2)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.attention = Attention(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro, data_clique):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch

        mol_n = mol_x.size(0)
        for i in range(len(self.mol_conv)):
            x = self.mol_conv[i](mol_x, mol_edge_index)
            if i < len(self.mol_conv) - 1:
                x = self.relu(x)
            if i == 0:
                mol_x = x
                continue
            mol_z = torch.sigmoid(
                self.mol_seq_fc1(x) + self.mol_seq_fc2(mol_x) + self.mol_bias.expand(mol_n, self.mol_out_feats))
            mol_x = mol_z * x + (1 - mol_z) * mol_x
        x = gmp(mol_x, mol_batch)  # global pooling
        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        pro_n = target_x.size(0)
        for i in range(len(self.pro_conv)):
            if i == 0:
                xt = self.pro_conv[i](target_x, target_edge_index, target_weight)
            else:
                xt = self.pro_conv[i](target_x, target_edge_index)
            if i < len(self.pro_conv) - 1:
                xt = self.relu(xt)
            if i == 0:
                target_x = xt
                continue
            pro_z = torch.sigmoid(
                self.pro_seq_fc1(xt) + self.pro_seq_fc2(target_x) + self.pro_bias.expand(pro_n, self.pro_out_feats))
            target_x = pro_z * xt + (1 - pro_z) * target_x
        xt = gmp(target_x, target_batch)  # global pooling
        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)
        # concat
        a = self.attention(x, xt)
        emb = torch.stack([x, xt], dim=1)
        a = a.unsqueeze(dim=2)
        emb = (a * emb).reshape(-1, 2 * 128)

        xc = self.fc1(emb)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class Graph_noConn(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, num_features_xc=92, embed_dim=128, output_dim=128, n_filters=32, dropout=0.2):
        super(Graph_noConn, self).__init__()

        self.n_output = n_output
        self.mol_conv = nn.ModuleList([])
        self.mol_conv.append(GATConv(num_features_mol, num_features_mol * 4, heads=2, dropout=dropout, concat=False))
        self.mol_conv.append(GATConv(num_features_mol * 4, num_features_mol * 4, heads=2, dropout=dropout, concat=False))
        self.mol_conv.append(GATConv(num_features_mol * 4, num_features_mol * 4, heads=2, dropout=dropout, concat=False))
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv = nn.ModuleList([])
        self.pro_conv.append(GCNConv(num_features_pro, num_features_pro * 4))
        self.pro_conv.append(GATConv(num_features_pro * 4, num_features_pro * 4, heads=2, dropout=dropout, concat=False))
        self.pro_conv.append(GATConv(num_features_pro * 4, num_features_pro * 4, heads=2, dropout=dropout, concat=False))
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.clique_conv = nn.ModuleList([])
        self.clique_conv.append(GATConv(num_features_xc, num_features_xc * 4, heads=2, dropout=dropout, concat=False))
        self.clique_conv.append(GATConv(num_features_xc * 4, num_features_xc * 4, heads=2, dropout=dropout, concat=False))
        self.clique_conv.append(GATConv(num_features_xc * 4, num_features_xc * 4, heads=2, dropout=dropout, concat=False))
        self.clique_fc_g1 = torch.nn.Linear(num_features_xc * 4, 1024)
        self.clique_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.attention = Attention(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(3 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro, data_clique):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        # get clique input
        clique_x, clique_edge_index, clique_batch = data_clique.x, data_clique.edge_index, data_clique.batch

        for i in range(len(self.mol_conv)):
            mol_x = self.mol_conv[i](mol_x, mol_edge_index)
            if i < len(self.mol_conv) - 1:
                mol_x = self.relu(mol_x)
        x = gmp(mol_x, mol_batch)  # global pooling
        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        for i in range(len(self.pro_conv)):
            if i == 0:
                target_x = self.pro_conv[i](target_x, target_edge_index, target_weight)
            else:
                target_x = self.pro_conv[i](target_x, target_edge_index)
            if i < len(self.pro_conv) - 1:
                target_x = self.relu(target_x)
        xt = gmp(target_x, target_batch)  # global pooling
        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # clique graph embedding
        for i in range(len(self.clique_conv)):
            clique_x = self.clique_conv[i](clique_x, clique_edge_index)
            if i < len(self.clique_conv) - 1:
                clique_x = self.relu(clique_x)
        xq = gmp(clique_x, clique_batch)  # global max pooling
        # flatten
        xq = self.relu(self.clique_fc_g1(xq))
        xq = self.dropout(xq)
        xq = self.clique_fc_g2(xq)
        xq = self.dropout(xq)
        # concat
        a = self.attention(x, xt, xq)
        emb = torch.stack([x, xt, xq], dim=1)
        a = a.unsqueeze(dim=2)
        emb = (a * emb).reshape(-1, 3 * 128)
        # add some dense layers
        xc = self.fc1(emb)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

class Graph_noATT(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, num_features_xc=92, embed_dim=128, output_dim=128, n_filters=32, dropout=0.2):
        super(Graph_noATT, self).__init__()

        print('DTA_GAT Loading ...')
        self.n_output = n_output
        self.mol_conv = nn.ModuleList([])
        self.mol_conv.append(GATConv(num_features_mol, num_features_mol *4, heads=2, dropout=dropout, concat=False))
        self.mol_conv.append(GATConv(num_features_mol* 4, num_features_mol* 4, heads=2, dropout=dropout, concat=False))
        self.mol_conv.append(GATConv(num_features_mol* 4, num_features_mol* 4, heads=2, dropout=dropout, concat=False))
        self.mol_out_feats = num_features_mol * 4
        self.mol_seq_fc1 = nn.Linear(num_features_mol*4, num_features_mol*4)
        self.mol_seq_fc2 = nn.Linear(num_features_mol*4, num_features_mol*4)
        self.mol_bias = nn.Parameter(torch.rand(1, num_features_mol*4))
        torch.nn.init.uniform_(self.mol_bias, a=-0.2, b=0.2)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv = nn.ModuleList([])
        self.pro_conv.append(GCNConv(num_features_pro, num_features_pro * 4))
        self.pro_conv.append(GATConv(num_features_pro * 4, num_features_pro * 4, heads=2, dropout=dropout, concat=False))
        self.pro_conv.append(GATConv(num_features_pro * 4, num_features_pro * 4, heads=2, dropout=dropout, concat=False))
        self.pro_out_feats = num_features_pro * 4
        self.pro_seq_fc1 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        self.pro_seq_fc2 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        self.pro_bias = nn.Parameter(torch.rand(1, num_features_pro * 4))
        torch.nn.init.uniform_(self.pro_bias, a=-0.2, b=0.2)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.clique_conv = nn.ModuleList([])
        self.clique_conv.append(GATConv(num_features_xc, num_features_xc * 4, heads=2, dropout=dropout, concat=False))
        self.clique_conv.append(GATConv(num_features_xc * 4, num_features_xc * 4, heads=2, dropout=dropout, concat=False))
        self.clique_conv.append(GATConv(num_features_xc * 4, num_features_xc * 4, heads=2, dropout=dropout, concat=False))
        self.clique_out_feats = num_features_xc * 4
        self.clique_seq_fc1 = nn.Linear(num_features_xc * 4, num_features_xc * 4)
        self.clique_seq_fc2 = nn.Linear(num_features_xc * 4, num_features_xc * 4)
        self.clique_bias = nn.Parameter(torch.rand(1, num_features_xc * 4))
        torch.nn.init.uniform_(self.clique_bias, a=-0.2, b=0.2)
        self.clique_fc_g1 = torch.nn.Linear(num_features_xc * 4, 1024)
        self.clique_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(3 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro, data_clique):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        # get clique input
        clique_x, clique_edge_index, clique_batch = data_clique.x, data_clique.edge_index,data_clique.batch

        mol_n = mol_x.size(0)
        for i in range(len(self.mol_conv)):
            x = self.mol_conv[i](mol_x, mol_edge_index)
            if i < len(self.mol_conv)-1:
                x = self.relu(x)
            if i==0:
                mol_x = x
                continue
            mol_z = torch.sigmoid(self.mol_seq_fc1(x) + self.mol_seq_fc2(mol_x) + self.mol_bias.expand(mol_n, self.mol_out_feats))
            mol_x = mol_z * x + (1 - mol_z) * mol_x

        x = gmp(mol_x, mol_batch)  # global pooling
        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        pro_n = target_x.size(0)
        for i in range(len(self.pro_conv)):
            if i == 0:
                xt = self.pro_conv[i](target_x, target_edge_index, target_weight)
            else:
                xt = self.pro_conv[i](target_x, target_edge_index)
            if i < len(self.pro_conv) - 1:
                xt = self.relu(xt)
            if i == 0:
                target_x = xt
                continue
            pro_z = torch.sigmoid(
                self.pro_seq_fc1(xt) + self.pro_seq_fc2(target_x) + self.pro_bias.expand(pro_n, self.pro_out_feats))
            target_x = pro_z * xt + (1 - pro_z) * target_x

        xt = gmp(target_x, target_batch)  # global pooling
        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # clique graph embedding
        xq_n = clique_x.size(0)
        for i in range(len(self.clique_conv)):
            xq = self.clique_conv[i](clique_x, clique_edge_index)
            if i < len(self.clique_conv) - 1:
                xq = self.relu(xq)
            if i == 0:
                clique_x = xq
                continue
            clique_z = torch.sigmoid(
                self.clique_seq_fc1(xq) + self.clique_seq_fc2(clique_x) + self.clique_bias.expand(xq_n, self.clique_out_feats))
            clique_x = clique_z * xq + (1 - clique_z) * clique_x

        xq = gmp(clique_x, clique_batch)  # global max pooling
        # flatten
        xq = self.relu(self.clique_fc_g1(xq))
        xq = self.dropout(xq)
        xq = self.clique_fc_g2(xq)
        xq = self.dropout(xq)
        # concat
        xc = torch.cat((x, xt, xq), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

