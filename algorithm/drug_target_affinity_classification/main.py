import os

from algorithm.drug_target_affinity_classification.models import DrugBAN
from algorithm.drug_target_affinity_classification.utils import graph_collate_func, drug_preprocess
from algorithm.drug_target_affinity_classification.configs import get_cfg_defaults
from algorithm.drug_target_affinity_classification.dataloader import DTIDataset
from torch.utils.data import DataLoader
from algorithm.drug_target_affinity_classification.predict import test
import torch
import numpy as np
import pdb
from rdkit import Chem


def drug_target_classification_prediction(drug_smiles, target_seq):
    """
    Predicts whether the drug will interact with the target, or the probability of interaction.

    Args:
        drug_smiles (list): A list of drug SMILES strings.
        target_seq (str): The target protein sequence. Never give '(user_target_seq)'.

    """


    model_type= 'model_DTI'
    batch_size=64
    device='cuda:0'

    if isinstance(drug_smiles, str):
        drug_smiles = [drug_smiles]
    if Chem.MolFromSmiles(drug_smiles[0]) is None:
        return("invalid smile")
    
    if not is_valid_protein_sequence(target_seq):
        if target_seq[-1] == '.':
            return "Error: target_seq should not end with '.'"
        return("This is an invalid target_seq, it cannot be predicted.")

    cur_path = os.path.dirname(__file__)

    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(f'{cur_path}/configs/DrugBAN_LL4.yaml')
    # print(f"Running on: {device}", end="\n")

    valid_drug_list, false_flag = drug_preprocess(drug_smiles)

    test_dataset = DTIDataset(valid_drug_list, target_seq)

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}
    params['shuffle'] = False
    params['drop_last'] = False

    test_generator = DataLoader(test_dataset, **params)

    model = DrugBAN(**cfg).to(device)

    torch.backends.cudnn.benchmark = True
    if model_type == 'model_DTI':
        PATH = f'{cur_path}/pretrained_models/model_DTI.model'
    model.load_state_dict(torch.load(PATH, map_location=device))
    result = test(model, device, test_generator)
    for i, flag in enumerate(false_flag):
        if not flag:
            result = np.insert(result, i, None)
    output = {}
    output['result'] = 'Finish predicting the probability that drug SMILES and target will interact'
    output['target_seq'] = target_seq
    # output['interact_probability'] = round(result[0], 2)
    output['result_values'] = []
    for i in range(len(drug_smiles)):
        cur_result = {}
        cur_result['smile'] = drug_smiles[i]
        cur_result['value'] = round(result[i], 2)
        output['result_values'].append(cur_result)
    return output

def is_valid_protein_sequence(sequence):
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    return all(residue in valid_amino_acids for residue in sequence)

if __name__ == '__main__':
    model_type = 'model_DTI'
    drug_smiles = [
        "CC(=O)Nc1cc(N)c(C#N)c(-c2ccccc2)n1"
    ]
    target_seq = ('MERKVLALQARKKRTKAKKDKAQRKSETQHRGSAPHSESDLPEQEEEILGSDDDEQEDPNDYCKGGYHLVKIGDLFNGRYHVIRKLGWGHFSTVWLSWDI'
                  'QGKKFVAMKVVKSAEHYTETALDEIRLLKSVRNSDPNDPNREMVVQLLDDFKISGVNGTHICMVFEVLGHHLLKWIIKSNYQGLPLPCVKKIIQQVLQGL'
                  'DYLHTKCRIIHTDIKPENILLSVNEQYIRRLAAEATEWQRSGAPPPSGSAVSTAPQPKPADKMSKNKKKKLKKKQKRQAELLEKRMQEIEEMEKESGPGQ'
                  'KRPNKQEESESPVERPLKENPPNKMTQEKLEESSTIGQDQTLMERDTEGGAAEINCNGVIEVINYTQNSNNETLRHKEDLHNANDCDVQNLNQESSFLSS'
                  'QNGDSSTSQETDSCTPITSEVSDTMVCQSSSTVGQSFSEQHISQLQESIRAEIPCEDEQEQEHNGPLDNKGKSTAGNFLVNPLEPKNAEKLKVKIADLGN'
                  'ACWVHKHFTEDIQTRQYRSLEVLIGSGYNTPADIWSTACMAFELATGDYLFEPHSGEEYTRDEDHIALIIELLGKVPRKLIVAGKYSKEFFTKKGDLKHI'
                  'TKLKPWGLFEVLVEKYEWSQEEAAGFTDFLLPMLELIPEKRATAAECLRHPWLNS')
    batch_size = 64
    device = 'cuda:0'

    result = drug_target_classification_prediction(drug_smiles=drug_smiles,
                                                   target_seq=target_seq)

    print(result)
