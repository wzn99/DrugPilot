import os.path
import pdb
import torch
from torch_geometric.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from algorithm.drug_target_affinity_regression.models.ColdDTA_model_clip import ColdDTA
# from .models.GraphDTA_model import GraphDTA
# from .models.MSF_DTA_model import MSF_DTA
# from .models.MSGNN_DTA_model import MSGNN_DTA
from algorithm.drug_target_affinity_regression.utils import *


def drug_target_affinity_regression_predict(drug_smiles, target_seq):
    """
    Predict the affinity between the drug SMILES and the target sequence.

    Args:
        drug_smiles (list): A list of drug SMILES strings.
        target_seq (str): The target protein sequence. Never give '(user_target_seq)'. 
    
    Example args:
        {'drug_smiles': 'CS(=O)(=O)N1CCN(Cc2cc3nc(-c4cccc5[nH]ncc45)nc(N4CCOCC4)c3s2)CC1', 'target_seq': 'MGCIKSKENKSPAIKYRPENT'}

    """


    batch_size=32
    device='cuda:0'
    model_type='ColdDTA'

    if isinstance(drug_smiles, str):
        drug_smiles = [drug_smiles]
        
    if Chem.MolFromSmiles(drug_smiles[0]) is None:
        return("invalid smile")
    
    if not is_valid_protein_sequence(target_seq):
        if target_seq[0] in ['(']:
            return f"Error: target_seq should not start with {target[0]}"
        if target_seq[-1] in ['.', ')']:
            return f"Error: target_seq should not end with {target[-1]}"
        return("This is an invalid target_seq, it cannot be predicted.")
    
    pred = -1

    total_preds = None
    false_flag = None
    if model_type == 'ColdDTA':
        model = ColdDTA(device=device).to(device)

        model_path = f"{os.path.dirname(__file__)}/pretrained_models/ColdDTA.model"
        # model_path = "pretrained_models/ColdDTA.model"
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        data_list, false_flag = preprocess_ColdDTA(drug_smiles, target_seq)
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
    # print(affinity_value)
    output = {}
    # output['result'] = round(affinity_value[0], 2)
    output['result'] = 'You have finished predicting the affinity regression value between the drug and the target sequence.'
    output['result_values'] = []
    for i in range(len(drug_smiles)):
        result = {}
        result['smile'] = drug_smiles[i]
        result['value'] = round(affinity_value[i], 2)
        output['result_values'].append(result)
    return output

def is_valid_protein_sequence(sequence):
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    return all(residue in valid_amino_acids for residue in sequence)

if __name__ == '__main__':
    # import os
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    import torch

    # 检查系统中可用的CUDA设备数量
    num_devices = torch.cuda.device_count()
    print("可用的CUDA设备数量：", num_devices)

    # 打印每个设备的信息
    for i in range(num_devices):
        print("CUDA 设备 {}: {}".format(i, torch.cuda.get_device_name(i)))
    smile = ['CC(C)n1cc2c3c(cccc31)[C@H]1C[C@@H](COC(=O)C3CC3)CN(C)[C@@H]1C2', 'CC(=O)Nc1cc(N)c(C#N)c(-c2ccccc2)n1']
    target = 'MEILCEDNISLSSIPNSLMQLGDGPRLYHNDFNSRDANTSEASNWTIDAENRTNLSCEGYLPPTCLSILHLQEKNWSALLTTVVIILTIAGNILVIMAVSLEKKLQNATNYFLMSLAIADMLLGFLVMPVSMLTILYGYRWPLPSKLCAIWIYLDVLFSTASIMHLCAISLDRYVAIQNPIHHSRFNSRTKAFLKIIAVWTISVGISMPIPVFGLQDDSKVFKEGSCLLADDNFVLIGSFVAFFIPLTIMVITYFLTIKSLQKEATLCVSDLSTRAKLASFSFLPQSSLSSEKLFQRSIHREPGSYAGRRTMQSISNEQKACKVLGIVFFLFVVMWCPFFITNIMAVICKESCNENVIGALLNVFVWIGYLSSAVNPLVYTLFNKTYRSAFSRYIQCQYKENRKPLQLILVNTIPALAYKSSQLQVGQKKNSQEDAEQTVDDCSMVTLGKQQSEENCTDNIETVNEKVSCV'

    device = 'cuda:3'

    # 示例输入
    input = {
        "drug_smiles": ["CC(=O)Nc1cc(N)c(C#N)c(-c2ccccc2)n1", "CC(C)n1cc2c3c(cccc31)[C@H]1C[C@@H](COC(=O)C3CC3)CN(C)[C@@H]1C2"],
        "target_seq":   'MERKVLALQARKKRTKAKKDKAQRKSETQHRGSAPHSESDLPEQEEEILGSDDDEQEDPNDYCKGGYHLVKIGDLFNGRYHVIRKLGWGHFSTVWLSWDI'
                        'QGKKFVAMKVVKSAEHYTETALDEIRLLKSVRNSDPNDPNREMVVQLLDDFKISGVNGTHICMVFEVLGHHLLKWIIKSNYQGLPLPCVKKIIQQVLQGL'
                        'DYLHTKCRIIHTDIKPENILLSVNEQYIRRLAAEATEWQRSGAPPPSGSAVSTAPQPKPADKMSKNKKKKLKKKQKRQAELLEKRMQEIEEMEKESGPGQ'
                        'KRPNKQEESESPVERPLKENPPNKMTQEKLEESSTIGQDQTLMERDTEGGAAEINCNGVIEVINYTQNSNNETLRHKEDLHNANDCDVQNLNQESSFLSS'
                        'QNGDSSTSQETDSCTPITSEVSDTMVCQSSSTVGQSFSEQHISQLQESIRAEIPCEDEQEQEHNGPLDNKGKSTAGNFLVNPLEPKNAEKLKVKIADLGN'
                        'ACWVHKHFTEDIQTRQYRSLEVLIGSGYNTPADIWSTACMAFELATGDYLFEPHSGEEYTRDEDHIALIIELLGKVPRKLIVAGKYSKEFFTKKGDLKHI'
                        'TKLKPWGLFEVLVEKYEWSQEEAAGFTDFLLPMLELIPEKRATAAECLRHPWLNS'
    }
    output = {}

    affinity_value = drug_target_affinity_regression_predict(drug_smiles=smile, target_seq=target)

    print(affinity_value)