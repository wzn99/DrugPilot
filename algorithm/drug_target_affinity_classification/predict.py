import torch
from algorithm.drug_target_affinity_classification.models import binary_cross_entropy
from tqdm import tqdm
import pdb

def test(model, device, test_dataloader):
    test_loss = 0
    y_label, y_pred = [], []

    data_loader = test_dataloader

    with torch.no_grad():
        model.eval()
        for i,(v_d, v_p, labels) in enumerate(data_loader):
            v_d, v_p, labels = v_d.to(device), v_p.to(device), labels.float().to(device)
            
            v_d, v_p, f, score, score_tw, class_dis = model(v_d, v_p)  #测试模型

            n, loss = binary_cross_entropy(class_dis, labels)
            test_loss += loss.item()
            y_label = y_label + labels.to("cpu").tolist()
            y_pred = y_pred + n.to("cpu").tolist()

    return y_pred

