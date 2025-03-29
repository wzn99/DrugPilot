import os
import pdb
import csv
from rdkit import Chem
from tqdm import tqdm
from algorithm.drug_drug_response.data_preprocessing import generate_drug_data, save_data_rewrite, load_data_statistics, \
    _corrupt_ent_rewrite, \
    _normal_batch_rewrite
from algorithm.drug_drug_response.ddi_datasets import CustomData, split_train_valid, TwosidesDataset, DrugDataset, \
    DrugDataLoader, \
    load_ddi_data_fold, initialize_drug_indices
import pandas as pd
import pickle
import numpy as np
import torch
from algorithm.drug_drug_response import models
import json


def validate_and_filter_pairs(smiles_pairs):
    valid_pairs = []
    invalid_indices = []

    for idx, (smiles1, smiles2) in enumerate(smiles_pairs):
        mol1 = Chem.MolFromSmiles(smiles1.strip())
        mol2 = Chem.MolFromSmiles(smiles2.strip())
        if mol1 is not None and mol2 is not None and smiles1.strip() != smiles2.strip():
            valid_pairs.append((smiles1, smiles2))
        else:
            invalid_indices.append(idx)

    return valid_pairs, invalid_indices


def generate_csv_data(smiles_pairs, relations):
    unique_smiles = list(set([smile for pair in smiles_pairs for smile in pair]))
    smiles_to_id = {smile: i for i, smile in enumerate(unique_smiles, start=1)}
    rows = []
    for smiles1, smiles2 in smiles_pairs:
        id1 = smiles_to_id[smiles1]
        id2 = smiles_to_id[smiles2]
        for relation in relations:
            rows.append([id1, id2, relation, smiles1, smiles2])
    header = ['ID1', 'ID2', 'Y', 'X1', 'X2']
    return pd.DataFrame(rows, columns=header)


def load_drug_mol_data(data, symbols):
    drug_id_mol_tup = []
    drug_smile_dict = {}

    for _, row in data.iterrows():
        drug_smile_dict[row['ID1']] = row['X1']
        drug_smile_dict[row['ID2']] = row['X2']

    for id, smiles in drug_smile_dict.items():
        mol = Chem.MolFromSmiles(smiles.strip())
        drug_id_mol_tup.append((id, mol))
    drug_data = {id: generate_drug_data(mol, symbols) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs')}
    return drug_data


def generate_pair_triplets(drug_data, data, dataset_type):
    pos_triplets = []
    drug_ids = list(drug_data.keys())

    for _, row in data.iterrows():
        id1, id2, relation = row['ID1'], row['ID2'], row['Y']
        if id1 in drug_ids and id2 in drug_ids:
            pos_triplets.append([id1, id2, relation])

    if not pos_triplets:
        raise ValueError('All tuples are invalid.')
    pos_triplets = np.array(pos_triplets)
    data_statistics = load_data_statistics(pos_triplets)
    drug_ids = np.array(drug_ids)

    neg_samples = []
    for pos_item in tqdm(pos_triplets, desc='Generating Negative sample'):
        h, t, r = pos_item[:3]
        if dataset_type == "drugbank":
            neg_heads, neg_tails = _normal_batch_rewrite(h, t, r, 1, data_statistics, drug_ids, seed=0)
            temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + [str(neg_t) + '$t' for neg_t in neg_tails]
        else:  # elif dataset_type == "twosides"
            existing_drug_ids = np.asarray(list(set(
                np.concatenate(
                    [data_statistics["ALL_TRUE_T_WITH_HR"][(h, r)], data_statistics["ALL_TRUE_H_WITH_TR"][(h, r)]],
                    axis=0)
            )))
            temp_neg = _corrupt_ent_rewrite(existing_drug_ids, 1, drug_ids, np.random.RandomState(0))
        neg_samples.append('_'.join(map(str, temp_neg[:1])))

    return pd.DataFrame({'Drug1_ID': pos_triplets[:, 0], 'Drug2_ID': pos_triplets[:, 1], 'Y': pos_triplets[:, 2],
                         'Neg samples': neg_samples})


def load_ddi_data_fold_rewrite(data, all_drug_data, batch_size, data_size_ratio, dataset_type):
    all_drug_data = {drug_id: CustomData(x=drug_data[0], edge_index=drug_data[1], edge_feats=drug_data[2],
                                         line_graph_edge_index=drug_data[3])
                     for drug_id, drug_data in all_drug_data.items()}

    initialize_drug_indices(all_drug_data)

    # Assuming 'data' DataFrame has columns 'Drug1_ID', 'Drug2_ID', 'Y', 'Neg samples'
    pos_triplets = [(d1, d2, r) for d1, d2, r in zip(data['Drug1_ID'], data['Drug2_ID'], data['Y'])]
    neg_samples = [[str(e) for e in neg_s.split('_')] for neg_s in data['Neg samples']]
    CustomDataset = DrugDataset if dataset_type == 'drugbank' else TwosidesDataset
    test_data = CustomDataset((np.array(pos_triplets), np.array(neg_samples)), all_drug_data, ratio=data_size_ratio,
                              seed=0)
    test_data_loader = DrugDataLoader(test_data, batch_size=batch_size)
    return test_data_loader


def do_compute(model, batch, device):
    batch = [t.to(device) for t in batch]
    p_score, n_score = model(batch)
    assert p_score.ndim == 2
    assert n_score.ndim == 3
    probas_pred = np.concatenate([torch.sigmoid(p_score.detach()).cpu().mean(dim=-1),
                                  torch.sigmoid(n_score.detach()).mean(dim=-1).view(-1).cpu()])
    ground_truth = np.concatenate([np.ones(p_score.shape[0]), np.zeros(n_score.shape[:2]).reshape(-1)])

    return p_score, n_score, probas_pred, ground_truth


def run_batch_rewrite(model, data_loader, device):
    probas_pred = []
    ground_truth = []

    for batch in data_loader:
        p_score, n_score, batch_probas_pred, batch_ground_truth = do_compute(model, batch, device)

        probas_pred.append(batch_probas_pred)
        ground_truth.append(batch_ground_truth)

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)
    pred = (probas_pred >= 0.5).astype(int)

    return pred, ground_truth


def pred_drugbank(smiles_pairs, device, batch_size):
    # Constants and Model Configuration
    dataset_type = "drugbank"
    data_size_ratio = 1
    hid_feats = 64
    rel_total = 86
    n_iter = 10
    dropout = 0
    NUM_FEATURES = 70
    NUM_EDGE_FEATURES = 6
    relations = list(range(rel_total))
    symbols = ['S', 'Ag', 'Sb', 'Al', 'Tc', 'Gd', 'Bi', 'Cl', 'Cu', 'Pt', 'Br', 'Mg', 'Cr', 'Hg', 'Zn', 'N', 'Ti', 'B',
               'Au', 'Sr', 'C', 'Li', 'Co', 'O', 'Fe', 'Ca', 'Se', 'Si', 'K', 'H', 'Ga', 'Na', 'I', 'La', 'As', 'P',
               'Ra', 'F']

    # Initialize and load the model
    model = models.GmpnnCSNetDrugBank(NUM_FEATURES, NUM_EDGE_FEATURES, hid_feats, rel_total, n_iter, dropout)
    model.load_state_dict(torch.load(f'{os.path.dirname(__file__)}/pretrained_models/best_drugbank_model.model',
                                     map_location=torch.device(device)),
                          strict=True)
    model.to(device)
    model.eval()

    # Validate and process drug pairs
    valid_pairs, invalid_indices = validate_and_filter_pairs(smiles_pairs)
    if not valid_pairs:
        return [['invalid inputs']] * len(smiles_pairs)  # Handle all invalid pairs case

    data = generate_csv_data(valid_pairs, relations)
    drug_data = load_drug_mol_data(data, symbols)
    triplets_data = generate_pair_triplets(drug_data, data, dataset_type)
    test_data_loader = load_ddi_data_fold_rewrite(triplets_data, drug_data, batch_size, data_size_ratio, dataset_type)

    # Model prediction
    with torch.no_grad():
        pred, _ = run_batch_rewrite(model, test_data_loader, device)
    pred = pred[:len(pred) // 2]  # Ignore the negative sample

    # drugbank side effect info
    se_info = pd.read_csv(f'{os.path.dirname(__file__)}/data/drugbank_side_effect_info.csv')
    se_name_dict = dict(zip(se_info['Interaction type'], se_info.Description))

    # Construct outcomes
    possible_reactions = [[] for _ in range(len(smiles_pairs))]
    reaction_index = 0
    for idx in range(len(smiles_pairs)):
        if idx in invalid_indices:
            possible_reactions[idx] = ['invalid inputs']
        elif reaction_index < len(valid_pairs):
            possible_reactions[idx] = [se_name_dict[(i % rel_total) + 1] for i, p in
                                       enumerate(pred[reaction_index * rel_total:(reaction_index + 1) * rel_total],
                                                 reaction_index * rel_total) if p == 1]
            reaction_index += 1

    return possible_reactions


def pred_twosides(smiles_pairs, device, batch_size):
    # Constants and Model Configuration
    dataset_type = "twosides"
    data_size_ratio = 1
    hid_feats = 64
    rel_total = 963
    n_iter = 10
    dropout = 0
    NUM_FEATURES = 54
    NUM_EDGE_FEATURES = 6
    relations = list(range(rel_total))
    symbols = ['B', 'Au', 'Cl', 'Ca', 'P', 'Al', 'C', 'Se', 'As', 'La', 'N', 'F', 'Na', 'Co', 'Gd', 'Pt', 'I', 'S',
               'Br', 'Li', 'O', 'K']

    # Initialize and load the model
    model = models.GmpnnCSNetTwosides(NUM_FEATURES, NUM_EDGE_FEATURES, hid_feats, rel_total, n_iter, dropout)
    model.load_state_dict(torch.load(f'{os.path.dirname(__file__)}/pretrained_models/best_twosides_model.model', map_location=torch.device(device)),
                          strict=True)
    model.to(device)
    model.eval()

    # Validate and process drug pairs
    valid_pairs, invalid_indices = validate_and_filter_pairs(smiles_pairs)
    if not valid_pairs:
        return [['invalid inputs']] * len(smiles_pairs)  # Handle all invalid pairs case

    data = generate_csv_data(valid_pairs, relations)
    drug_data = load_drug_mol_data(data, symbols)
    triplets_data = generate_pair_triplets(drug_data, data, dataset_type)
    test_data_loader = load_ddi_data_fold_rewrite(triplets_data, drug_data, batch_size, data_size_ratio, dataset_type)

    # Model prediction
    with torch.no_grad():
        pred, _ = run_batch_rewrite(model, test_data_loader, device)
    pred = pred[:len(pred) // 2]  # Ignore the negative sample

    # TWOSIDES side effect info
    se_info = pd.read_csv(f'{os.path.dirname(__file__)}/data/twosides_side_effect_info.csv', index_col=0)
    se_name_dict = dict(zip(se_info.SE_map, se_info['Side Effect Name']))

    # Construct outcomes
    possible_reactions = [[] for _ in range(len(smiles_pairs))]
    reaction_index = 0
    for idx in range(len(smiles_pairs)):
        if idx in invalid_indices:
            possible_reactions[idx] = ['invalid inputs']
        elif reaction_index < len(valid_pairs):
            possible_reactions[idx] = [se_name_dict[i % rel_total] for i, p in
                                       enumerate(pred[reaction_index * rel_total:(reaction_index + 1) * rel_total],
                                                 reaction_index * rel_total) if p == 1]
            reaction_index += 1

    return possible_reactions


def drug_drug_response_predict(smiles_pairs):
    """
    Predicts the result of the interaction between two drugs, including potential side reactions.

    Args:
        smiles_pairs (list of list): A two-dimensional list of drug SMILES pairs, where each inner list contains two drug SMILES strings.

    Example args:
        'smiles_pairs': [['COC1=C2OC(=O)C=CC2=CC2=C1OC=C2', 'CC2=C1OC=C2']]
    """

    # 假数据
    # output = {}
    # output['result'] = 'Finished predicting the result of the interaction between two drugs'
    # output['result_values'] = [{'smile1':smiles_pairs[0][0], 'smile2':smiles_pairs[0][0], 'interact_result':'#Drug1 may increase the neurotoxic activities of #Drug2.'}]
    # return output

    device='cuda:0'
    batch_size=256

    if isinstance(smiles_pairs, str):
        smiles_pairs = json.loads(smiles_pairs)
        smiles_pairs = [smiles_pairs]
        
    # Initialize a list to hold results for each pair, with sublists for each dataset type
    results = [[] for _ in smiles_pairs]

    # Loop over dataset types and populate results accordingly
    for dataset_type in task_list:
        if dataset_type == 'drugbank':
            predictions = pred_drugbank(smiles_pairs, device, batch_size)
        else:  # Assume 'twosides'
            predictions = pred_twosides(smiles_pairs, device, batch_size)

        # Add predictions to each corresponding drug pair result list
        for i, prediction in enumerate(predictions):
            # Convert list of strings to a single string if there's more than one string
            formatted_prediction = '; '.join(prediction)
            results[i].append(formatted_prediction)

    # print(results)
    # Pair each smiles pair with its corresponding results and format output
    drug_pairs_and_results = []
    for smiles_pair, result in zip(smiles_pairs, results):
        drug_pairs_and_results.append([smiles_pair[0], smiles_pair[1], *result])
    drug_pairs_and_results.insert(0, ['drug1', 'drug2'] + task_list)

    output = {}
    output['result'] = 'Finished predicting the result of the interaction between two drugs'
    output['result_values'] = []
    for i in range(1, len(drug_pairs_and_results)):
        result = {}
        result['smile1'] = drug_pairs_and_results[i][0]
        result['smile2'] = drug_pairs_and_results[i][1]
        result['interact_result'] = ' '.join(drug_pairs_and_results[i][2:])
        output ['result_values'].append(result)
    return output


if __name__ == '__main__':
    # Sample input data
    smiles_pairs1 = [["[H]\\C(=C(\\[H])C1=NCCCN1C)C1=CC=CS1","[H][C@@]1(C[C@@](N)(CC2=C1C(O)=C1C(=O)C3=CC=CC=C3C(=O)C1=C2O)C(C)=O)O[C@H]1C[C@H](O)[C@H](O)CO1"]]
    smiles_pairs2 = [
        ["COC1=C2OC(=O)C=CC2=CC2=C1OC=C2", "COC(=O)CCC1=C2NC(\\C=C3/N=C(/C=C4\\N\\C(=C/C5=N/C(=C\\2)/C(CCC(O)=O)=C5C)C(C=C)=C4C)C(\\C(=O)OC)=C3C)=C1C"],
        ["InvalidSMILE", "C[C@H](CC1=CC=CC=C1)N(C)CC#C"]
    ]
    smiles_pairs3 = [
        ["COC1=C2OC(=O)C=CC2=CC2=C1OC=C2", "COC(=O)CCC1=C2NC(\\C=C3/N=C(/C=C4\\N\\C(=C/C5=N/C(=C\\2)/C(CCC(O)=O)=C5C)C(C=C)=C4C)C(\\C(=O)OC)=C3C)=C1C"],
        ["InvalidSMILE", "C[C@H](CC1=CC=CC=C1)N(C)CC#C"],
        ["CN1CCC(CC1)=C1C2=C(CCC3=CC=CC=C13)SC=C2", "C[C@H](CC1=CC=CC=C1)N(C)CC#C"]
    ]
    task_list = ['drugbank', 'twosides']
    device = 'cuda:0'
    batch_size = 256
    print(drug_drug_response_predict(smiles_pairs1, task_list))

    """
    [['drug1', 'drug2', 'drugbank', 'twosides'],
     ['COC1=C2OC(=O)C=CC2=CC2=C1OC=C2', 'COC(=O)CCC1=C2NC(\\C=C3/N=C(/C=C4\\N\\C(=C/C5=N/C(=C\\2)/C(CCC(O)=O)=C5C)C(C=C)=C4C)C(\\C(=O)OC)=C3C)=C1C', '#Drug1 may increase the myopathic rhabdomyolysis activities of #Drug2.; The bioavailability of #Drug2 can be increased when combined with #Drug1.', ''], 
     ['InvalidSMILE', 'C[C@H](CC1=CC=CC=C1)N(C)CC#C', 'invalid inputs', 'invalid inputs'], 
     ['CN1CCC(CC1)=C1C2=C(CCC3=CC=CC=C13)SC=C2', 'C[C@H](CC1=CC=CC=C1)N(C)CC#C', '', 'lung adenocarcinoma; polyneuropathy; ear discharge; Anal fistula; cheilosis; mouth pain; Dyspnea exertional']]
    """
