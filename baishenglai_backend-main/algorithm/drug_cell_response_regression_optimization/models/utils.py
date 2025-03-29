import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch,pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import pdb
from torch.utils.data.dataset import ConcatDataset

import random
class DRPDataset(InMemoryDataset):
    def __init__(self, dataset_path=None, root = 'data/',  transform=None,
                 pre_transform=None,smile_graph=None):
        #root is required for save preprocessed data, default is '/tmp'
        super(DRPDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.data, self.slices = torch.load(dataset_path)

            
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass
    
    def process(self, xd, xt, y,smile_graph):
        pass
    
def load(config):
    # config is total config not a sub config of 'dataset_type'
    
    
    # if config['select_type'] =='r3':
    #     data_list = []
    #     for pt_item in tqdm(os.listdir(config['dataset_path'])[:int(len(os.listdir(config['dataset_path']))*config['scale'])]):
            
    #         pt_data = DRPDataset(dataset_path = os.path.join(config['dataset_path'],pt_item))
    #         data_list.append(pt_data)

    #     train_size = int(config['train']*len(data_list))
    #     val_size = int(config['val']*len(data_list))+train_size

    #     random.shuffle(data_list)
        
    #     train_data = ConcatDataset(data_list[:train_size])
    #     val_data = ConcatDataset(data_list[train_size:val_size])
    #     test_data = ConcatDataset(data_list[val_size:])
        
    #     return train_data, val_data, test_data
    
    # elif config['select_type'] =='m2r':
        
    #     data_list = []
    #     for pt_item in tqdm(os.listdir(config['dataset_path'])[:int(len(os.listdir(config['dataset_path']))*config['scale'])]):
            
    #         pt_data = DRPDataset(dataset_path = os.path.join(config['dataset_path'],pt_item))
    #         data_list.append(pt_data)

    #     train_size = int(config['train']*len(data_list))
    #     val_size = int(config['val']*len(data_list))+train_size

    #     random.shuffle(data_list)
    #     trainval_data = ConcatDataset(data_list[:val_size])
    #     test_data = ConcatDataset(data_list[val_size:])
    #     train_data, val_data = torch.utils.data.random_split(trainval_data,\
    #         [
    #             int(len(trainval_data)*(config['train']/(config['train']+config['val']))),\
    #             len(trainval_data)-int(len(trainval_data)*(config['train']/(config['train']+config['val'])))
    #         ])
      
    #     return train_data, val_data, test_data
    
    # elif config['select_type'] =='s3':
        
    #     data_list = []
    #     for pt_item in tqdm(os.listdir(config['dataset_path'])[:int(len(os.listdir(config['dataset_path']))*config['scale'])]):
            
    #         pt_data = DRPDataset(dataset_path = os.path.join(config['dataset_path'],pt_item))
    #         data_list.append(pt_data)
        
    #     train_size = int(config['train']*len(data_list))
    #     val_size = int(config['val']*len(data_list))+train_size

    #     train_data = ConcatDataset(data_list[:train_size])
        
    #     test_data = ConcatDataset(data_list[val_size:])
        
    #     return train_data, test_data
    
    # elif config['select_type'] == '1f1':

    #     data_list = []
    #     # pt_items = os.listdir(config['dataset_path'])[:int(len(os.listdir(config['dataset_path']))*config['scale'])]
    #     for pt_item in tqdm(os.listdir(config['dataset_path'])[:int(len(os.listdir(config['dataset_path']))*config['scale'])]):
            
    #         pt_data = DRPDataset(dataset_path = os.path.join(config['dataset_path'],pt_item))
    #         data_list.append(pt_data)
    #     train_size = int(config['train']*len(data_list))
    #     # pt_data1 = DRPDataset(dataset_path = os.path.join(config['dataset_path'],'294734.pt'))
    #     pt_data2 = DRPDataset(dataset_path = os.path.join(config['dataset_path'],'282752.pt'))
        
    #     train_data = ConcatDataset(data_list[:train_size])
    #     test_data = ConcatDataset([pt_data2])
        
    #     return train_data, test_data

    # elif config['select_type'] =='many_lh':
    #     # 就是给出n个药物子数据集
    #     data_name_list = ['282752.pt','294734.pt', '711193.pt', '267033.pt', '269142.pt', '637731.pt', '645151.pt', '33530.pt', '618759.pt','634784.pt','642947.pt','618201.pt','617976.pt','618332.pt','641245.pt','805338.pt','805062.pt','805068.pt','804785.pt','177383.pt','83292.pt','174005.pt','348401.pt','174177.pt','383336.pt','687353.pt','656243.pt','20534.pt','720623.pt','643862.pt','68091.pt','673794.pt','40341.pt','651079.pt','645984.pt','665364.pt','625624.pt','46061.pt','685988.pt','100046.pt','617128.pt','686349.pt','312033.pt','7522.pt','101088.pt','637732.pt','645808.pt','638646.pt','246131.pt','665640.pt','645828.pt','641175.pt','645832.pt','286628.pt','645817.pt','627168.pt','641174.pt','639498.pt','645819.pt','640500.pt','266068.pt','631521.pt','637693.pt','363997.pt','692391.pt','613662.pt','639511.pt','643027.pt','677617.pt','648322.pt','634785.pt','264054.pt','634786.pt','626482.pt','91874.pt','626875.pt','678634.pt','678636.pt','800058.pt','800018.pt','817909.pt','781084.pt','812695.pt','405159.pt','782154.pt','801372.pt','817117.pt','817918.pt','819215.pt','109154.pt','17474.pt','819214.pt','802113.pt','799334.pt','24692.pt','801416.pt','803417.pt','820324.pt','60898.pt','817910.pt','817390.pt','818336.pt','817272.pt','817903.pt','801012.pt','802536.pt','816429.pt','332837.pt','799659.pt','828319.pt','3840.pt','802769.pt','796054.pt','205098.pt','833900.pt','811100.pt','782135.pt','810564.pt','797394.pt','819606.pt','799358.pt','67477.pt','813671.pt','67478.pt','820304.pt','797583.pt','796216.pt','367413.pt','812381.pt','810520.pt','48888.pt','783628.pt','797650.pt','819748.pt','816184.pt','811263.pt','818557.pt','819747.pt','783099.pt','817908.pt','801984.pt','802545.pt','796237.pt','783090.pt','816969.pt','783641.pt','47384.pt','781085.pt','813650.pt','100777.pt','810443.pt','787772.pt','367089.pt','831270.pt','220334.pt','816475.pt','84063.pt','798263.pt','788727.pt','813499.pt','787854.pt','128578.pt','793460.pt','800846.pt','89400.pt','128187.pt','57715.pt','2805.pt','813501.pt','819173.pt','317605.pt','221265.pt','13151.pt','794946.pt','793141.pt','813500.pt','803428.pt','37888.pt','780883.pt','811879.pt','69830.pt','780882.pt','78325.pt','804094.pt','58329.pt','30846.pt','781440.pt','800782.pt','677646.pt','636934.pt','228082.pt','813502.pt','57864.pt','277184.pt','797734.pt','815942.pt','676944.pt','795138.pt','715455.pt','812201.pt','2186.pt','796018.pt','812947.pt','131513.pt','156939.pt','143102.pt','57759.pt','798262.pt','813644.pt','808307.pt','787978.pt']
    #     data_list = {}
    #     for pt_item in tqdm(os.listdir(config['dataset_path'])[:int(len(os.listdir(config['dataset_path']))*config['scale'])]):
    #     # for pt_item in data_name_list:
    #         pt_data = DRPDataset(dataset_path = os.path.join(config['dataset_path'],pt_item))
    #         data_list[pt_data.smiles[0]]=pt_data
    #     return data_list
    
    
    # elif config['select_type'] =='manyGDSCv2':
    #     # 就是给出n个药物子数据集
    #     data_list = []
        # for pt_item in tqdm(os.listdir(config['dataset_path'])[:int(len(os.listdir(config['dataset_path']))*config['scale'])]):
        #     pt_data = DRPDataset(dataset_path = os.path.join(config['dataset_path'],pt_item))
        #     data_list.append(pt_data)
        # return data_list

    if config['dataset_type']['select_type'] =='zeroshot':
        data_list_train = []
        data_list_test = []
        if config['dataset_name'] == 'GDSCv2':
            test_smiles_pt = ['Belinostat', 'XMD8-85', 'GSK319347A', 'Tamoxifen', 'QL-XII-47', 'IOX2', 'Piperlongumine', 'PF-562271', 'Y-39983', 'Alectinib', 'Ponatinib', 'TAK-715',
                    'JW-7-24-1', 'Vinblastine', 'GW-2580', 'Zibotentan', 'Sepantronium bromide', 'Cytarabine', '5-Fluorouracil', 'Navitoclax', 'Rucaparib', 'JNK-9L', 'Pelitinib']
            for pt_item in tqdm(os.listdir(config['dataset_type']['dataset_path'])[:int(len(os.listdir(config['dataset_type']['dataset_path']))*config['dataset_type']['scale'])]):
                pt_data = DRPDataset(dataset_path = os.path.join(config['dataset_type']['dataset_path'],pt_item))
                
                if pt_item.split('.')[0] in test_smiles_pt:
                    data_list_test.append(pt_data)
                else:
                    data_list_train.append(pt_data)
                    
            train_data = ConcatDataset(data_list_train)
            test_data = ConcatDataset(data_list_test)
            return train_data, test_data
        
        elif config['dataset_name'] == 'CellMiner':
            test_smiles_pt = np.load("/home/lk/project/MSDA/data/process/NCI60_dataset/NCI60_drug_smiles_test.npy")
            for pt_item in tqdm(os.listdir(config['dataset_type']['dataset_path'])[:int(len(os.listdir(config['dataset_type']['dataset_path']))*config['dataset_type']['scale'])]):
                
                pt_data = DRPDataset(dataset_path = os.path.join(config['dataset_type']['dataset_path'],pt_item))

                if len(pt_data[0].smiles)>=290:
                    print(pt_data[0].smiles)
                    continue
                
                if pt_data[0].smiles in test_smiles_pt:
                    
                    
                    data_list_test.append(pt_data)
                else:
                    data_list_train.append(pt_data)
                    
            train_data = ConcatDataset(data_list_train)
            test_data = ConcatDataset(data_list_test)
            return train_data, test_data
    
def get_dict_smiles2pt(config):
    data_map = {}
    cell_names_map = {}
    
    cell_name_idx = 0
    for pt_item in tqdm(os.listdir(config['dataset_path'])[:int(len(os.listdir(config['dataset_path']))*config['scale'])]):
        pt_data = DRPDataset(dataset_path = os.path.join(config['dataset_path'],pt_item))

        data_map[pt_data.data.smiles[0]] = pt_item
        for item in pt_data.data.cell_name:
            if cell_names_map.get(item) is None:
                cell_names_map[item] = cell_name_idx
                cell_name_idx += 1
    return data_map,cell_names_map

def load_dataset_from_smiles(config, smiles):
    return DRPDataset(dataset_path = os.path.join(config['dataset_path'],smiles))
  
def copyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件



def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def draw_loss(train_losses, test_losses, title):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(title+".png")  # should before show method
    plt.close()
    
def draw_pearson(pearsons, title):
    plt.figure()
    plt.plot(pearsons, label='test pearson')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(title+".png")  # should before show method
    plt.close()
    
def draw_sort_pred_gt(pred,gt,title):
    # gt = gt.y.cpu().numpy()
    # pred = pred.squeeze().cpu().detach().numpy()
    # zipped = zip(gt,pred)
    # sort_zipped = sorted(zipped,key=lambda x:(x[0]))
    # data_gt, data_pred = [list(x) for x in zip(*sort_zipped)] 
    data_gt, data_pred = zip(*sorted(zip(gt,pred)))
    plt.figure()
    plt.scatter( np.arange(len(data_gt)),data_gt, s=0.1, alpha=1, label='gt')
    plt.scatter( np.arange(len(data_gt)),data_pred, s=0.1, alpha=1, label='pred')
    plt.legend()
    plt.savefig(title+".png")
    plt.close()

def draw_sort_pred_gt_classed(pred,gt,title,dataloader,marker):
    save_img = True
    
    drug = {}
    cell = {}
    
    idx = 0
    for batch_idx, data in tqdm(enumerate(dataloader)):
        
        
        for item_index in range(len(data)):
            item = data[item_index]
        
            if drug.get(item['drug_name']):
                drug[item['drug_name']].append({
                    'cell':item['cell_name'],
                    'IC50':gt[idx],
                    'Pred':pred[idx]
                    })
            else:
                drug[item['drug_name']] = []
                drug[item['drug_name']].append({
                    'cell':item['cell_name'],
                    'IC50':gt[idx],
                    'Pred':pred[idx]
                    })
                
            if cell.get(item['cell_name']):
                cell[item['cell_name']].append({
                    'drug':item['drug_name'],
                    'IC50':gt[idx],
                    'Pred':pred[idx]
                    })
            else:
                cell[item['cell_name']] = []
                cell[item['cell_name']].append({
                    'drug':item['drug_name'],
                    'IC50':gt[idx],
                    'Pred':pred[idx]
                    })
                
            if idx>= len(gt):
                print("Error!")
                break
            else:
                idx += 1
                
    if os.path.exists(title+'/drug_'+marker) is False:
        os.mkdir(title+'/drug_'+marker)
        
    if os.path.exists(title+'/cell_'+marker) is False:
        os.mkdir(title+'/cell_'+marker)

    result_drug = [0,0,0,0,0]
    result_cell = [0,0,0,0,0]
    rankingLossFunc = torch.nn.MarginRankingLoss(margin=0.0, reduction='mean')
    
    result_drugs = {}
    
    for drug_item in tqdm(drug):
        
        drug_data = drug[drug_item]
        
        D_gt = np.array([x['IC50'] for x in drug_data])
        D_pred = np.array([x['Pred'] for x in drug_data])
        
        result = [
                rmse(D_gt, D_pred),
                mse(D_gt, D_pred),
                pearson(D_gt, D_pred),
                spearman(D_gt, D_pred),
                rankingLossFunc(torch.tensor(D_gt) , torch.tensor(D_pred), torch.ones_like(torch.tensor(D_pred))).item()
            ]

        if np.nan not in result:
            result_drug = [i + j for i, j in zip(result_drug, result)]
            
        result = [ str(x)[:6] for x in result]
        
        result_drugs[drug_item] = result
        
        if save_img:
            data_gt, data_pred = zip(*sorted(zip(D_gt,D_pred)))
            
            plt.figure()
            plt.title(str(drug_item)+"_"+str(result),fontsize='10',fontweight='heavy') 
            plt.scatter( np.arange(len(data_gt)),data_gt, s=1, alpha=0.9, label='gt')
            plt.scatter( np.arange(len(data_gt)),data_pred, s=1, alpha=0.9, label='pred')
            plt.legend()
            drug_item = drug_item.replace("/","")
            plt.savefig(title+"/drug_"+marker+"/"+str(drug_item)+".png")
            plt.close()

    print(marker,"\tDrug:",np.divide(result_drug,len(drug)))
    print(marker,"\tCell:",np.divide(result_cell,len(cell)))
    
    result_drugs["Drug"] = [str(x)[:6] for x in np.divide(result_drug,len(drug))]
    return result_drugs


def num2english(num, PRECISION=3):
    num = str(round(num,PRECISION)).split('.')[1]
    
    while len(num)!=PRECISION:
        num = num + '0'

    L1 = ["zero","one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    word = ""
    for i in str(num):
        # pdb.set_trace()
        word= word+" "+L1[int(i)]
   
    return word


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss