import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
from algorithm.drug_target_affinity_classification.ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
import pdb

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1) #删除1维
    loss = loss_fct(n, labels)           #计算二元交叉熵损失
    #print(loss)
    return n, loss                       #n表示输出的类别


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]   #只取第二列
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))  #NLLLoss函数计算损失，取了平均损失
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class DrugBAN(nn.Module):
    def __init__(self, **config):
        super(DrugBAN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        #self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)
        self.protein_extractor = Protein_Transformer(protein_emb_dim, protein_padding)  #transformer进行靶蛋白表征
        #self.protein_extractor2 = Protein_Transformer(protein_emb_dim, protein_padding)  #transformer进行靶蛋白表征

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        #self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

        # fake孪生网络
        self.bcn2 = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        #self.mlp_classifier2 = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

        in_dim = 256
        hidden_dim = 512
        out_dim = 256
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, out_binary)

        #ZHZ 2023/12/3 改进 双通道输入，计算MLP输出层之间距离        
    def forward(self, bg_d, v_protein, mode="train"):
        v_p = self.protein_extractor(v_protein)     #靶点表征矩阵[batch, TargetLength-15, Out_DIM]


    
        v_d = self.drug_extractor(bg_d)             #药物表征矩阵[batch, nodeNum, Out_DIM]
        f_dt, att = self.bcn(v_d, v_p)

        f_tw, att_tw = self.bcn2(v_p, v_p)       #计算靶点和靶点的输出向量，使用相同的靶点
        #score_tw = self.mlp_classifier2(f_tw)        #计算水与靶点蛋白的联合表征结果   



        #计算之间的距离
        euclid_dist = f_dt - f_tw

        #pdb.set_trace()

        score_tw = 0
        score1 = score_tw

        euclid_dist = self.bn1(F.relu(self.fc1(euclid_dist)))
        euclid_dist = self.bn2(F.relu(self.fc2(euclid_dist)))
        euclid_dist = self.bn3(F.relu(self.fc3(euclid_dist)))
        euclid_dist = self.fc4(euclid_dist)
        #===========================================
        if mode == "train":
            return v_d, v_p, f_dt, score1, score_tw, euclid_dist
        elif mode == "eval":
            return v_d, v_p, score1, att, score_tw, euclid_dist
#化合物编码网络
class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')       #？？？？
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

#蛋白质编码网络
class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters   #[128, 128, 128, 128]
        self.in_ch = in_ch[-1]
        kernels = kernel_size  #[3, 6, 9]
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long()) # [batch_size,seq_len,embedding_size]
        v = v.transpose(2, 1)        # [batch_size,embedding_size,seq_len] 扫最后一维
        v = self.bn1(F.relu(self.conv1(v)))  
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)  # v.size(2)=seq_len-2-5-8
        return v

class Protein_Transformer(nn.Module):
    def __init__(self, embedding_dim, padding=True):
        super(Protein_Transformer, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        #in_ch = [embedding_dim] + num_filters   #[128, 128, 128, 128]
        
        #transformer编码层
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8), num_layers=6)
        #self.PE = PositionalEncoding(embedding_dim,0.1,max_len=1200)

    def forward(self, v):
        v = self.embedding(v.long()) # [batch_size,seq_len,embedding_size]
        #v = self.PE(v) 
        v = self.transformer_encoder(v)
        return v

# 2.位置编码层  2023.12.19
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=1200):
        '''
        :param d_model: 词嵌入维度
        :param dropout: 丢失率
        :param max_len: 每个句子的最长长度
        '''
        super(PositionalEncoding,self).__init__()
        # 实例化dropout层
        self.dpot = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len,d_model)

        # 初始化绝对位置矩阵
        # position矩阵size为(max_len,1)
        position = torch.arange(0,max_len).unsqueeze(1)

        # 将绝对位置矩阵和位置编码矩阵特征融合
        # 定义一个变换矩阵 跳跃式初始化
        div_term = torch.exp(torch.arange(0,d_model,2) * -(math.log(10000)/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        # 将二维张量扩充成三维张量
        pe = pe.unsqueeze((0))

        # 把pe位置编码矩阵注册成模型的buffer
        # 模型保存后重加载时和模型结构与参数一同被加载
        self.register_buffer('pe',pe)

    def forward(self,x):
        '''
        :param x: 文本的词嵌入表示
        :return:
        '''

        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dpot(x)


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
