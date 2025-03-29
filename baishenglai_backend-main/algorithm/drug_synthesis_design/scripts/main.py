import torch
import sys
sys.path.append('./algorithm/drug_synthesis_design')
sys.path.append('./algorithm/drug_synthesis_design/scripts')
from tqdm import tqdm
from functools import partial
from algorithm.drug_synthesis_design.scripts.utils import *
from dgl.data.utils import save_graphs, load_graphs
from get_edit import get_bg_partition, mask_prediction, combined_edit
from Decode_predictions import read_prediction, decode_localtemplate
from rdkit import Chem

def is_valid_smile(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        return mol is not None
    except:
        return False

class USPTOTestDataset(object):
    def __init__(self, smile_list, args, smiles_to_graph, node_featurizer, edge_featurizer, load=True, log_every=1000):
        self.smiles = smile_list
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer, load, log_every)

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer, load, log_every):
        print('Processing test dgl graphs from scratch...')
        self.graphs = []
        for i, s in enumerate(self.smiles):
            if (i + 1) % log_every == 0:
                print('Processing molecule %d/%d' % (i+1, len(self.smiles)))
            self.graphs.append(smiles_to_graph(s, node_featurizer=node_featurizer,
                                               edge_featurizer=edge_featurizer, canonical_atom_order=False))

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item], None

    def __len__(self):
        return len(self.smiles)
    
def load_testloader(smile_list, args):
    test_set = USPTOTestDataset(smile_list, args, 
                        smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'])
    test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs_test, num_workers=0)
    return test_loader

def write_edits(args, model, test_loader):
    model.eval()
    results = []
    with torch.no_grad():
        for batch_id, data in enumerate(test_loader):
            smiles_list, bg, _ = data
            batch_atom_logits, batch_bond_logits, _ = predict(args, model, bg)    
            sg = bg.remove_self_loop()
            graphs = dgl.unbatch(sg)
            batch_atom_logits = nn.Softmax(dim = 1)(batch_atom_logits)
            batch_bond_logits = nn.Softmax(dim = 1)(batch_bond_logits) 
            graphs, nodes_sep, edges_sep = get_bg_partition(bg)
            start_node = 0
            start_edge = 0
            print('\rProcessing test molecule batch %s/%s' % (batch_id, len(test_loader)), end='', flush=True)
            for single_id, (graph, end_node, end_edge) in enumerate(zip(graphs, nodes_sep, edges_sep)):
                smiles = smiles_list[single_id]
                test_id = (batch_id * args['batch_size']) + single_id
                atom_logits = batch_atom_logits[start_node:end_node]
                bond_logits = batch_bond_logits[start_edge:end_edge]
                atom_logits, bond_logits = mask_prediction(smiles, atom_logits, bond_logits, args['site_templates'])
                pred_types, pred_sites, pred_scores = combined_edit(graph, atom_logits, bond_logits, args['top_num'])
                start_node = end_node
                start_edge = end_edge
                results.append((test_id, smiles, [(pred_types[i], pred_sites[i][0], pred_sites[i][1], pred_scores[i]) for i in range(args['top_num'])]))

    return results


def decode_predictions(args, raw_predictions):
    base_dir = './algorithm/drug_synthesis_design'
    atom_templates = pd.read_csv(f'{base_dir}/data/%s/atom_templates.csv' % args['dataset'])
    bond_templates = pd.read_csv(f'{base_dir}/data/%s/bond_templates.csv' % args['dataset'])
    template_infos = pd.read_csv(f'{base_dir}/data/%s/template_infos.csv' % args['dataset'])
    args['rxn_class_given'] = False
    args['atom_templates'] = {atom_templates['Class'][i]: atom_templates['Template'][i] for i in atom_templates.index}
    args['bond_templates'] = {bond_templates['Class'][i]: bond_templates['Template'][i] for i in bond_templates.index}
    args['template_infos'] = {template_infos['Template'][i]: {'edit_site': eval(template_infos['edit_site'][i]), 'change_H': eval(template_infos['change_H'][i]), 'change_C': eval(template_infos['change_C'][i]), 'change_S': eval(template_infos['change_S'][i])} for i in template_infos.index}
    args['raw_predictions'] = {test_id: [smiles] + preds for test_id, smiles, preds in raw_predictions}

    result_dict = {}
    partial_func = partial(get_k_predictions, args=args)
    
    tasks = range(len(raw_predictions))
    for task_id in tqdm(tasks, total=len(tasks), desc='Decoding LocalRetro predictions'):
        result = partial_func(task_id)
        result_dict[result[0]] = result[1]

    # Create the header dynamically based on top_num
    header = ["product_smile"]
    for i in range(1, args['top_num'] + 1):
        header.append(f"reactant{i}")
        header.append(f"score{i}")
    
    decoded_predictions = [header]  # Add the header as the first element
    for i in sorted(result_dict.keys()):
        all_prediction, _ = result_dict[i]
        formatted_prediction = [args['raw_predictions'][i][0]]  # Start with product_smile
        for reactant_score in all_prediction:
            reactant, score = eval(reactant_score)
            formatted_prediction.append(reactant)
            formatted_prediction.append(score)
        decoded_predictions.append(formatted_prediction)
        print('\rDecoding LocalRetro predictions %d/%d' % (i, len(raw_predictions)), end='', flush=True)
    print()

    return decoded_predictions

def get_k_predictions(test_id, args):
    raw_prediction = args['raw_predictions'][test_id]
    all_prediction = []
    class_prediction = []
    product = raw_prediction[0]
    predictions = raw_prediction[1:]
    for prediction in predictions:
        mol, pred_site, template, template_info, score = read_prediction(product, prediction, args['atom_templates'], args['bond_templates'], args['template_infos'], True)
        local_template = '>>'.join(['(%s)' % smarts for smarts in template.split('_')[0].split('>>')])
        decoded_smiles = decode_localtemplate(mol, pred_site, local_template, template_info)
        try:
            decoded_smiles = decode_localtemplate(mol, pred_site, local_template, template_info)
            if decoded_smiles == None or str((decoded_smiles, score)) in all_prediction:
                continue
        except Exception as e:
#                     print (e)
            continue
        all_prediction.append(str((decoded_smiles, score)))

        if args['rxn_class_given']:
            rxn_class = args['test_rxn_class'][test_id]
            if template in args['templates_class'][str(rxn_class)].values:
                class_prediction.append(str((decoded_smiles, score)))
            if len (class_prediction) >= args['top_num']:
                break

        elif len (all_prediction) >= args['top_num']:
            break
    return (test_id, (all_prediction, class_prediction))

def Retrosynthetic_reaction_pathway_prediction(smiles_list):
    """
    Predicts the possible precursors of drug molecules that could react and transform into the given drug molecules (SMILES).

    Args:
        smiles_list (list): A list of drug SMILES strings for which the retrosynthetic precursors are to be predicted.

    """

    # 假数据
    # output = {}
    # output['result'] = 'Finished predicting the possible precursors of drug molecules that could react and transform into the given molecules'
    # output['result_values'] = ['FCCl.OC(C(F)(F)F)C(F)(F)F']
    # return output
    
    dataset='USPTO_50K'
    top_k_result=3
    device='cuda:0'
    batch_size=16

    args = {}
    args['mode'] = 'test'
    args['model'] = 'default'
    args['device'] = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
    print('Using device %s' % args['device'])
    args['dataset'] = dataset
    model_name = 'LocalRetro_%s.model' % dataset
    base_dir = './algorithm/drug_synthesis_design'
    args['model_path'] = f'{base_dir}/pretrained_models/%s' % model_name
    args['config_path'] = f'{base_dir}/data/configs/default_config.json'
    args['data_dir'] = f'{base_dir}/data/{dataset}'
    args['top_num'] = top_k_result
    args['batch_size'] = batch_size

    args = init_featurizer(args)
    args = get_site_templates(args)
    model = load_model(args)

    # Check validity of SMILES
    valid_smiles = []
    invalid_smiles_results = {}
    for idx, smile in enumerate(smiles_list):
        if is_valid_smile(smile):
            valid_smiles.append((idx, smile))
        else:
            invalid_smiles_results[idx] = [smile, 'invalid input!']
            return 'contains invalid smile, it cannot be predicted'
    
    valid_smile_list = [smile for idx, smile in valid_smiles]
    if valid_smile_list:
        test_loader = load_testloader(valid_smile_list, args)
        raw_predictions = write_edits(args, model, test_loader)
        predictions = decode_predictions(args, raw_predictions)
        valid_smile_results = {idx: predictions[i+1] for i, (idx, smile) in enumerate(valid_smiles)}  # Skip header in predictions
    else:
        valid_smile_results = {}

    # Create the header dynamically based on top_num
    header = ["product_smile"]
    for i in range(1, args['top_num'] + 1):
        header.append(f"reactant{i}")
        header.append(f"score{i}")
    # Combine valid and invalid results and keep the original order
    final_results = [header]
    for idx in range(len(smiles_list)):
        if idx in invalid_smiles_results:
            final_results.append(invalid_smiles_results[idx])
        else:
            final_results.append(valid_smile_results[idx])
    reactants = final_results[1][1]
    output = {}
    output['result'] = 'Finished predicting the possible precursors of drug molecules that could react and transform into the given molecules'
    output['result_values'] = []
    for i in range(1, len(final_results)):
        result = {}
        result['smile'] = final_results[i][0]  
        result['reactants'] = final_results[i][1: len(final_results[i]): 2]  
        output['result_values'].append(result)
        # for j in range(1, len(final_results[i]), 2):
        #     reactants = final_results[i][1]  
        #     # 使用 . 进行分割
        #     reactant_parts = reactants.split('.')
        #     # 分配给 reactant1 和 reactant2
        #     result['reactant1'] = reactant_parts[0]  # . 前的部分
        #     result['reactant2'] = reactant_parts[1]  # . 后的部分
        #     output['result'].append(result)
    # print(final_results)
    return output

if __name__ == '__main__':
    smile_list = ['CC(C)NCC(COC1=CC=CC2=CC=CC=C21)O', 'B([C@H](CC(C)C)NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)(O)O']
    # dataset = 'USPTO_50K'  # or USPTO_MIT
    # top_k_result = 3
    # device = 'cuda:0'
    # batch_size = 16
    result = Retrosynthetic_reaction_pathway_prediction(smile_list)
    print(result)
    '''
    result = 
    [['product_smile', 'reactant1', 'score1', 'reactant2', 'score2', 'reactant3', 'score3'], 
    ['C(OC(C(F)(F)F)C(F)(F)F)F', 'FCCl.OC(C(F)(F)F)C(F)(F)F', 0.58525646, 'FCCl.[O-]C(C(F)(F)F)C(F)(F)F', 0.009981233], 
    ['C1=NC(=NN1C2C(C(C(O2)CO)O)O)C(=O)N', 'COC(=O)c1ncn(C2OC(CO)C(O)C2O)n1.N', 0.5999637, 'CCOC(=O)c1ncn(C2OC(CO)C(O)C2O)n1.N', 0.33185047, 'NC(=O)c1nc[nH]n1.OCC1OC(Cl)C(O)C1O', 0.051176306], 
    ['invalid_smile', 'invalid input!'], 
    ['CC(C)NCC(COC1=CC=CC2=CC=CC=C21)O', 'CC(C)N.OC(CCl)COc1cccc2ccccc12', 0.47017166, 'CC(C)N.OC(CBr)COc1cccc2ccccc12', 0.09113783, 'CC(C)N.OCC(O)COc1cccc2ccccc12', 0.0682635]]
    '''