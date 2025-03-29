import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp

# GCN based model
class MSF_DTA(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2):

        super(MSF_DTA, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_prot = nn.Dropout(0.0)

        #protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        self.dense = nn.Linear(800,512)
        self.dense_ = nn.Linear(512,256)
        self.dense__ = nn.Linear(256, 128)

        self.dense1 = nn.Linear(167, 128)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)#2*output_dim
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch,fingerprint,prot_seq = data.x, data.edge_index, data.batch,data.fingerprint,data.prot_seq
        # get protein input
        target = data.target

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        # embedded_xt = self.embedding_xt(prot_seq)
        # conv_xt = self.conv_xt_1(embedded_xt)
        # # flatten
        # xt_seq = conv_xt.view(-1, 32 * 121)
        # xt_seq = self.fc1_xt(xt_seq)

        xt = self.dense(target)

        xt = self.relu(xt)
        xt = self.dropout_prot(xt)
        xt = self.dense_(xt)

        xt = self.relu(xt)
        xt = self.dropout_prot(xt)
        xt = self.dense__(xt)

        fingerprint = self.dense1(fingerprint)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        out = self.out(xc)
        return out
