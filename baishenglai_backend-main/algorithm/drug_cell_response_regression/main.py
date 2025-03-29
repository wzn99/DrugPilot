import os.path
from torch_geometric.data import DataLoader
from algorithm.drug_cell_response_regression.models.TransEDRP import TransEDRP
from algorithm.drug_cell_response_regression.utils import *
import math


def drug_cell_response_regression_predict(drug_smiles, cell_name):
    """
    Predict the interaction value (z-score) between the drug SMILES and the cell line.

    Args:
        drug_smiles (list): A list of drug SMILES strings.
        cell_name (Any): The name of the cell line.

    """


    batch_size=128
    device='cuda:0'
    model_type='TransEDRP'

    if isinstance(drug_smiles, str):
        drug_smiles = [drug_smiles]
    if Chem.MolFromSmiles(drug_smiles[0]) is None:
        return("invalid smile")
    try:
        pred = -1
        cur_path = os.path.dirname(__file__)

        total_preds = None
        false_flag = None
        if model_type == 'TransEDRP':
            model = TransEDRP().to(device)

            model_path = f"{cur_path}/pretrained_models/TransEDRP.model"
            target_file_path = f"{cur_path}/PANCANCER_Genetic_feature.csv"

            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            data_list, false_flag = preprocess_DRP(drug_smiles, cell_name, target_file_path=target_file_path)
            data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, drop_last=False)

            model.eval()
            total_preds = torch.Tensor()
            for batch_idx, data in enumerate(data_loader):
                data = data.to(device)
                pred = model(data)
                pred = pred.cpu().view(-1, 1)
                total_preds = torch.cat((total_preds, pred), 0)
            total_preds = total_preds.detach().numpy().flatten()

        affinity_value = total_preds
        for idx, flag in enumerate(false_flag):
            if not flag:
                affinity_value = np.insert(affinity_value, idx, None)
        
        output = {}
        output['result'] = 'Finished predicting the interaction value (z-score) between the drug SMILES and the cell line.'
        output['cell_line'] = cell_name
        output['result_values'] = []
        for i in range(len(drug_smiles)):
            result = {}
            ic50 = affinity_value[i]
            zscore = -10 * math.log(1 / ic50 - 1) # ic50变换为zscore
            zscore_rounded = round(zscore, 2) # 保存两位小数
            result['smile'] = drug_smiles[i]
            result['value'] = round(zscore_rounded, 2)
            output['result_values'].append(result)
        return output
    except KeyError as e:
        return 'invalid cell_name'

if __name__ == '__main__':
    # import os
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    import torch

    # 检查系统中可用的CUDA设备数量
    # num_devices = torch.cuda.device_count()
    # print("可用的CUDA设备数量：", num_devices)

    # # 打印每个设备的信息
    # for i in range(num_devices):
    #     print("CUDA 设备 {}: {}".format(i, torch.cuda.get_device_name(i)))

    # drug_smiles = [
    #     'CC(=O)Nc1cc(N)c(C#N)c(-c2ccccc2)n1',
    #     'COc1cc2c(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3F)ccnc2cc1OCCCN1CCOCC1',
    #     'N#CCC(C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1.O=P(O)(O)O',
    #     '00000',  # 错误示例测试
    #     'CNCNCOOOCCCCC',
    #     '',  # 错误示例测试
    #     '6',  # 错误示例测试
    #     'CC(C)(C)c1cc(NC(=O)9999c2ccc(-c3cn4c(n3)sc3cc(OCCN5CCOCC5)NNNc34)cc2)no1'  # 错误示例测试
    # ]
    drug_smiles = ['CC(=O)Nc1cc(N)c(C#N)c(-c2ccccc2)n1', 'CC(=O)Nc1cc(N)c(C#N)c(-c2ccccc2)n1']
    cell_name = ('1290730')
    affinity_value = drug_cell_response_regression_predict(drug_smiles=drug_smiles,
                                                           cell_name=cell_name)
    print(affinity_value)
