import torch.optim as optim
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from model import CombinedScoreNetwork
from models.transE import TransE

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

import pandas as pd
from dataset import DrugDataset
from tqdm import tqdm
import random
import numpy as np
import logging
from datetime import datetime
import os

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    

def train(model, transe, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = [0, 0, 0, 0]
    cross_entropy_loss_1 = torch.nn.CrossEntropyLoss()
    cross_entropy_loss_2 = torch.nn.CrossEntropyLoss()
    cross_entropy_loss_3 = torch.nn.CrossEntropyLoss()
    cross_entropy_loss_4 = torch.nn.CrossEntropyLoss()

    for index, data in tqdm(enumerate(dataloader)):
        data = data.to(device)

        optimizer.zero_grad()
        # score_x, score_adj = model(data)
        
        logits_per_dc, logits_per_text, num_logits_per_dc, num_logits_per_text, y_pred = model(data)
        
        number_features = model.generate_samples(0, 1000)
        
        loss_TransEs = transe(number_features, number_features)
        
        loss_TransE = torch.sum(loss_TransEs[0]) + torch.sum(loss_TransEs[1]) + torch.sum(loss_TransEs[2])
    
        labels = torch.arange(data.ic50.shape[0]).long().to(device)

        loss_dc = cross_entropy_loss_1(logits_per_dc, labels)
        loss_t = cross_entropy_loss_2(logits_per_text, labels)
        loss_dc_num = cross_entropy_loss_3(num_logits_per_dc, labels)
        loss_t_num = cross_entropy_loss_4(num_logits_per_text, labels)
        
        loss_CLIP = (loss_dc + loss_t) / 2 
        loss_CLIP_Num = (loss_dc_num + loss_t_num) / 2 
        
        loss = loss_TransE * 0.0001 + loss_CLIP * 0.9 + loss_CLIP_Num * 0.1
        # loss = loss_CLIP
        
        loss.backward()
        optimizer.step()
        
        total_loss[0] += loss.item()
        total_loss[1] += loss_TransE.item()
        total_loss[2] += loss_CLIP.item()
        total_loss[3] += loss_CLIP_Num.item()

        logger.info(f"Iter[{index+1}], Step train loss: {total_loss[0]/(index+1)}, TransE loss: {total_loss[1]/(index+1)}, CLIP loss: {total_loss[2]/(index+1)}, CLIP Num loss: {total_loss[3]/(index+1)}")
        
        # 手动释放不再需要的变量
        del data, logits_per_dc, logits_per_text, num_logits_per_dc, num_logits_per_text, y_pred, number_features, labels, loss_TransEs
        torch.cuda.empty_cache()
        
    return [loss_item / len(dataloader) for loss_item in total_loss]

def evaluate(model, transe, dataloader, criterion, device):
    model.eval()
    total_loss = [0, 0, 0, 0]
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for index, data in tqdm(enumerate(dataloader)):

            data = data.to(device)
            logits_per_dc, logits_per_text, num_logits_per_dc, num_logits_per_text, y_pred = model(data)
            number_features = model.generate_samples(0, 1000)
        
            loss_TransEs = transe(number_features, number_features)
            
            loss_TransE = torch.sum(loss_TransEs[0]) + torch.sum(loss_TransEs[1]) + torch.sum(loss_TransEs[2])
        
            labels = torch.arange(data.ic50.shape[0]).long().to(device)

            loss_dc = cross_entropy_loss(logits_per_dc, labels)
            loss_t = cross_entropy_loss(logits_per_text, labels)
            loss_dc_num = cross_entropy_loss(num_logits_per_dc, labels)
            loss_t_num = cross_entropy_loss(num_logits_per_text, labels)
            
            loss_CLIP = (loss_dc + loss_t) / 2 
            loss_CLIP_Num = (loss_dc_num + loss_t_num) / 2 
            
            loss = loss_TransE * 0.0001 + loss_CLIP * 0.9 + loss_CLIP_Num * 0.1
            # loss = loss_CLIP
            
            total_loss[0] += loss.item()
            total_loss[1] += loss_TransE.item()
            total_loss[2] += loss_CLIP.item()
            total_loss[3] += loss_CLIP_Num.item()

            logger.info(f"Iter[{index+1}], Step valid loss: {total_loss[0]/(index+1)}, TransE loss: {total_loss[1]/(index+1)}, CLIP loss: {total_loss[2]/(index+1)}, CLIP Num loss: {total_loss[3]/(index+1)}")

            # 手动释放不再需要的变量
            del data, logits_per_dc, logits_per_text, num_logits_per_dc, num_logits_per_text, y_pred, number_features, labels, loss_TransEs
            torch.cuda.empty_cache()

    return [loss_item / len(dataloader) for loss_item in total_loss]

# 配置日志记录
def setup_logger(task_name, current_time):
    log_filename = f"./log/{task_name}_training_log_{current_time}.log"
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

seed_torch(42)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger(task_name='CLDR_Pretrain', current_time=current_time)

# 读取CSV文件
csv_file = '/home/lk/project/repaint/MolPaint/data/gdscv2.csv'
df = pd.read_csv(csv_file)

# 创建数据集
dataset = DrugDataset(root='/home/lk/project/repaint/MolPaint/data', df=df)

# 分割数据集为训练集、验证集和测试集
train_idx, val_test_idx = train_test_split(list(range(len(dataset))), test_size=0.3, random_state=42)
train_dataset = dataset[train_idx]
val_dataset = dataset[val_test_idx]

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# 定义模型
model = CombinedScoreNetwork(device, max_feat_num=10, max_node_num=100, nhid=64, num_layers=3, num_linears=2, 
                             c_init=2, c_hid=64, c_final=1, adim=64, num_heads=4, conv='GCN').to(device)

cell_csv_path = '/home/lk/project/repaint/MolPaint/data/PANCANCER_Genetic_feature.csv'
model.get_cell_matrix(cell_csv_path)

transe = TransE(1000, 1, device, dim=128)
transe = transe.to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.05)
criterion = torch.nn.MSELoss()

# 早停配置
k = 5  # 设置连续多少次没有改善时停止训练
best_epoch = -1
best_val_loss = float('inf')
epochs_no_improve = 0

# 训练模型
num_epochs = 50

for epoch in tqdm(range(num_epochs)):
    train_loss = train(model, transe, train_loader, optimizer, criterion, device)
    logger.info(f"Epoch[{epoch}] train loss: {train_loss[0]}, TransE loss: {train_loss[1]}, CLIP loss: {train_loss[2]}, CLIP Num loss: {train_loss[3]}")

    val_loss = evaluate(model, transe, val_loader, criterion, device)
    logger.info(f"Epoch[{epoch}] valid loss: {val_loss[0]}, TransE loss: {val_loss[1]}, CLIP loss: {val_loss[2]}, CLIP Num loss: {val_loss[3]}")

    # 检查是否有新的最优验证损失
    if val_loss[0] < best_val_loss:
        best_val_loss = val_loss[0]
        best_epoch = epoch + 1
        epochs_no_improve = 0

        # 保存最优模型和训练配置
        model_save_path = f"./pth/CL_best_model_{current_time}.pt"
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': {
                'num_epochs': num_epochs,
                # 其他训练配置
            }
        }, model_save_path)

        logger.info(f"New best model saved at epoch {epoch + 1} with val loss {best_val_loss:.4f}")

    else:
        epochs_no_improve += 1

    # 检查是否达到早停条件
    if epochs_no_improve == k:
        logger.info(f"Early stopping at epoch {epoch + 1}")
        logger.info(f"Best model was at epoch {best_epoch} with val loss {best_val_loss:.4f}")
        break
