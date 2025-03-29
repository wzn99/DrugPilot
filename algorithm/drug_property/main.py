import sys

import numpy as np
import yaml
from algorithm.drug_property.utils.util import *
from algorithm.drug_property.utils.dataset_loader import *
from algorithm.drug_property.models.model_single_cls import GraphMultiHead as Model_CLS
from algorithm.drug_property.models.model_single_reg import GraphMultiHead as Model_REG
import torch
from tqdm import tqdm
from random import shuffle
from copy import deepcopy
from torch_geometric.data import Data
from rdkit import Chem
from torch.utils.data import Dataset


def convert_smiles_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    _, x, edge_index = mol_to_graph(mol)
    # 为PyTorch Geometric创建Data实例
    data = Data(x=x,
                edge_index=edge_index,
                smile=smile)
    return data


class DrugDataset(Dataset):
    def __init__(self, smiles_list):
        # Convert smiles to graphs
        self.data = [convert_smiles_to_graph(smiles) for smiles in smiles_list]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(smiles_list, task):
    valid_smiles_list = []
    valid_indices = []
    for index, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:  # filter out None entries
            valid_smiles_list.append(smiles)
            valid_indices.append(index)
    return DrugDataset(valid_smiles_list), valid_indices


def predicting(model, device, loader, loader_type, args):
    model.eval()
    total_preds = torch.Tensor()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            pred, _ = model(data)
            total_preds = torch.cat((total_preds, pred.cpu().view(-1, 1)), 0)

    return total_preds.numpy().flatten()


def drug_property_prediction(drug_smiles, property):
    """
    Predicting the value of the specified property of the specified drug SMILES.

    Args:
        drug_smiles (list): A list of drug SMILES strings.
        property (str): The property to predict. Must be one of the following:
            - 'Class': Predicts the inhibitory activity on BACE1.
            - 'p_np': Predicts the blood-brain barrier permeability.
            - 'logSolubility': Predicts the water solubility.
            - 'freesolv': Predicts the free energy.
            - 'lipo': Predicts the lipid solubility.
            
    """


    batch_size=32
    device='cuda:0'

    # 把列表字符串转为列表
    import ast
    try:
        parsed = ast.literal_eval(drug_smiles)
        if isinstance(parsed, list):
            drug_smiles = parsed
    except (ValueError, SyntaxError):
        pass  # 如果解析失败，保持原样
    
    # 把输入的字符串参数转为列表
    if isinstance(drug_smiles, str):
        drug_smiles = [drug_smiles]
    # import pdb; pdb.set_trace()
    if Chem.MolFromSmiles(drug_smiles[0]) is None or len(drug_smiles[0]) == 1:
        return("This is an invalid SMILE, it's property cannot be predicted.")
    
    if isinstance(property, list):
        property = property[0]
    if isinstance(property, str) and property.startswith('(') and property.endswith(')'):
        property = property[1:-1]
    task = get_task(property)
    if task is None:
        return "This is an invalid property, this property cannot be predicted." 
    task_list = [task]

    cls_tasks = ['BACE', 'BBBP', 'ClinTox', 'SIDER', 'Tox21', 'ToxCast']
    reg_tasks = ['ESOL', 'FreeSolv', 'LIPO']
    task_descriptions = {
        'BACE': 'Finished predicting the probability this drug can inhibit β-secretase (BACE-1) which is an enzyme closely related to the development of Alzheimer\'s disease',
        'BBBP': 'Finished predicting the probability that the compound can effectively penetrate the blood-brain barrier, which is very important for the design of central nervous system drugs.',
        'ESOL': 'Finished predicting the magnitude of the drug\'s water solubility. Water solubility is an important factor in drug design, especially in terms of the drug\'s bioavailability and efficacy.',
        'FreeSolv': 'Finished predicting the magnitude of the solvation free energy of the compound in water, which is crucial for understanding the behavior of molecules in biological systems.',
        'LIPO': 'Finished predicting the magnitude of the compound\'s lipophilicity (logP). Lipophilicity is an important parameter for predicting how a drug passes through cell membranes.'
    }
    task_properties = {
        'BACE': ['Class'],
        'BBBP': ['p_np'],
        'ClinTox': ['FDA_APPROVED', 'CT_TOX'],
        'SIDER': ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
                  'Investigations', 'Musculoskeletal and connective tissue disorders', 'Gastrointestinal disorders',
                  'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders',
                  'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                  'General disorders and administration site conditions', 'Endocrine disorders',
                  'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders',
                  'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders',
                  'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders',
                  'Psychiatric disorders', 'Renal and urinary disorders',
                  'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders', 'Cardiac disorders',
                  'Nervous system disorders', 'Injury, poisoning and procedural complications'],
        'Tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
                  'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
        'ToxCast': ['ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive', 'APR_HepG2_CellCycleArrest_24h_dn',
                    'APR_HepG2_CellCycleArrest_24h_up', 'APR_HepG2_CellCycleArrest_72h_dn', 'APR_HepG2_CellLoss_24h_dn',
                    'APR_HepG2_CellLoss_72h_dn', 'APR_HepG2_MicrotubuleCSK_24h_dn', 'APR_HepG2_MicrotubuleCSK_24h_up',
                    'APR_HepG2_MicrotubuleCSK_72h_dn', 'APR_HepG2_MicrotubuleCSK_72h_up', 'APR_HepG2_MitoMass_24h_dn',
                    'APR_HepG2_MitoMass_24h_up', 'APR_HepG2_MitoMass_72h_dn', 'APR_HepG2_MitoMass_72h_up',
                    'APR_HepG2_MitoMembPot_1h_dn', 'APR_HepG2_MitoMembPot_24h_dn', 'APR_HepG2_MitoMembPot_72h_dn',
                    'APR_HepG2_MitoticArrest_24h_up', 'APR_HepG2_MitoticArrest_72h_up', 'APR_HepG2_NuclearSize_24h_dn',
                    'APR_HepG2_NuclearSize_72h_dn', 'APR_HepG2_NuclearSize_72h_up', 'APR_HepG2_OxidativeStress_24h_up',
                    'APR_HepG2_OxidativeStress_72h_up', 'APR_HepG2_StressKinase_1h_up', 'APR_HepG2_StressKinase_24h_up',
                    'APR_HepG2_StressKinase_72h_up', 'APR_HepG2_p53Act_24h_up', 'APR_HepG2_p53Act_72h_up',
                    'APR_Hepat_Apoptosis_24hr_up', 'APR_Hepat_Apoptosis_48hr_up', 'APR_Hepat_CellLoss_24hr_dn',
                    'APR_Hepat_CellLoss_48hr_dn', 'APR_Hepat_DNADamage_24hr_up', 'APR_Hepat_DNADamage_48hr_up',
                    'APR_Hepat_DNATexture_24hr_up', 'APR_Hepat_DNATexture_48hr_up', 'APR_Hepat_MitoFxnI_1hr_dn',
                    'APR_Hepat_MitoFxnI_24hr_dn', 'APR_Hepat_MitoFxnI_48hr_dn', 'APR_Hepat_NuclearSize_24hr_dn',
                    'APR_Hepat_NuclearSize_48hr_dn', 'APR_Hepat_Steatosis_24hr_up', 'APR_Hepat_Steatosis_48hr_up',
                    'ATG_AP_1_CIS_dn', 'ATG_AP_1_CIS_up', 'ATG_AP_2_CIS_dn', 'ATG_AP_2_CIS_up', 'ATG_AR_TRANS_dn',
                    'ATG_AR_TRANS_up', 'ATG_Ahr_CIS_dn', 'ATG_Ahr_CIS_up', 'ATG_BRE_CIS_dn', 'ATG_BRE_CIS_up',
                    'ATG_CAR_TRANS_dn', 'ATG_CAR_TRANS_up', 'ATG_CMV_CIS_dn', 'ATG_CMV_CIS_up', 'ATG_CRE_CIS_dn',
                    'ATG_CRE_CIS_up', 'ATG_C_EBP_CIS_dn', 'ATG_C_EBP_CIS_up', 'ATG_DR4_LXR_CIS_dn',
                    'ATG_DR4_LXR_CIS_up', 'ATG_DR5_CIS_dn', 'ATG_DR5_CIS_up', 'ATG_E2F_CIS_dn', 'ATG_E2F_CIS_up',
                    'ATG_EGR_CIS_up', 'ATG_ERE_CIS_dn', 'ATG_ERE_CIS_up', 'ATG_ERRa_TRANS_dn', 'ATG_ERRg_TRANS_dn',
                    'ATG_ERRg_TRANS_up', 'ATG_ERa_TRANS_up', 'ATG_E_Box_CIS_dn', 'ATG_E_Box_CIS_up', 'ATG_Ets_CIS_dn',
                    'ATG_Ets_CIS_up', 'ATG_FXR_TRANS_up', 'ATG_FoxA2_CIS_dn', 'ATG_FoxA2_CIS_up', 'ATG_FoxO_CIS_dn',
                    'ATG_FoxO_CIS_up', 'ATG_GAL4_TRANS_dn', 'ATG_GATA_CIS_dn', 'ATG_GATA_CIS_up', 'ATG_GLI_CIS_dn',
                    'ATG_GLI_CIS_up', 'ATG_GRE_CIS_dn', 'ATG_GRE_CIS_up', 'ATG_GR_TRANS_dn', 'ATG_GR_TRANS_up',
                    'ATG_HIF1a_CIS_dn', 'ATG_HIF1a_CIS_up', 'ATG_HNF4a_TRANS_dn', 'ATG_HNF4a_TRANS_up',
                    'ATG_HNF6_CIS_dn', 'ATG_HNF6_CIS_up', 'ATG_HSE_CIS_dn', 'ATG_HSE_CIS_up', 'ATG_IR1_CIS_dn',
                    'ATG_IR1_CIS_up', 'ATG_ISRE_CIS_dn', 'ATG_ISRE_CIS_up', 'ATG_LXRa_TRANS_dn', 'ATG_LXRa_TRANS_up',
                    'ATG_LXRb_TRANS_dn', 'ATG_LXRb_TRANS_up', 'ATG_MRE_CIS_up', 'ATG_M_06_TRANS_up', 'ATG_M_19_CIS_dn',
                    'ATG_M_19_TRANS_dn', 'ATG_M_19_TRANS_up', 'ATG_M_32_CIS_dn', 'ATG_M_32_CIS_up', 'ATG_M_32_TRANS_dn',
                    'ATG_M_32_TRANS_up', 'ATG_M_61_TRANS_up', 'ATG_Myb_CIS_dn', 'ATG_Myb_CIS_up', 'ATG_Myc_CIS_dn',
                    'ATG_Myc_CIS_up', 'ATG_NFI_CIS_dn', 'ATG_NFI_CIS_up', 'ATG_NF_kB_CIS_dn', 'ATG_NF_kB_CIS_up',
                    'ATG_NRF1_CIS_dn', 'ATG_NRF1_CIS_up', 'ATG_NRF2_ARE_CIS_dn', 'ATG_NRF2_ARE_CIS_up',
                    'ATG_NURR1_TRANS_dn', 'ATG_NURR1_TRANS_up', 'ATG_Oct_MLP_CIS_dn', 'ATG_Oct_MLP_CIS_up',
                    'ATG_PBREM_CIS_dn', 'ATG_PBREM_CIS_up', 'ATG_PPARa_TRANS_dn', 'ATG_PPARa_TRANS_up',
                    'ATG_PPARd_TRANS_up', 'ATG_PPARg_TRANS_up', 'ATG_PPRE_CIS_dn', 'ATG_PPRE_CIS_up', 'ATG_PXRE_CIS_dn',
                    'ATG_PXRE_CIS_up', 'ATG_PXR_TRANS_dn', 'ATG_PXR_TRANS_up', 'ATG_Pax6_CIS_up', 'ATG_RARa_TRANS_dn',
                    'ATG_RARa_TRANS_up', 'ATG_RARb_TRANS_dn', 'ATG_RARb_TRANS_up', 'ATG_RARg_TRANS_dn',
                    'ATG_RARg_TRANS_up', 'ATG_RORE_CIS_dn', 'ATG_RORE_CIS_up', 'ATG_RORb_TRANS_dn', 'ATG_RORg_TRANS_dn',
                    'ATG_RORg_TRANS_up', 'ATG_RXRa_TRANS_dn', 'ATG_RXRa_TRANS_up', 'ATG_RXRb_TRANS_dn',
                    'ATG_RXRb_TRANS_up', 'ATG_SREBP_CIS_dn', 'ATG_SREBP_CIS_up', 'ATG_STAT3_CIS_dn', 'ATG_STAT3_CIS_up',
                    'ATG_Sox_CIS_dn', 'ATG_Sox_CIS_up', 'ATG_Sp1_CIS_dn', 'ATG_Sp1_CIS_up', 'ATG_TAL_CIS_dn',
                    'ATG_TAL_CIS_up', 'ATG_TA_CIS_dn', 'ATG_TA_CIS_up', 'ATG_TCF_b_cat_CIS_dn', 'ATG_TCF_b_cat_CIS_up',
                    'ATG_TGFb_CIS_dn', 'ATG_TGFb_CIS_up', 'ATG_THRa1_TRANS_dn', 'ATG_THRa1_TRANS_up', 'ATG_VDRE_CIS_dn',
                    'ATG_VDRE_CIS_up', 'ATG_VDR_TRANS_dn', 'ATG_VDR_TRANS_up', 'ATG_XTT_Cytotoxicity_up',
                    'ATG_Xbp1_CIS_dn', 'ATG_Xbp1_CIS_up', 'ATG_p53_CIS_dn', 'ATG_p53_CIS_up', 'BSK_3C_Eselectin_down',
                    'BSK_3C_HLADR_down', 'BSK_3C_ICAM1_down', 'BSK_3C_IL8_down', 'BSK_3C_MCP1_down', 'BSK_3C_MIG_down',
                    'BSK_3C_Proliferation_down', 'BSK_3C_SRB_down', 'BSK_3C_Thrombomodulin_down',
                    'BSK_3C_Thrombomodulin_up', 'BSK_3C_TissueFactor_down', 'BSK_3C_TissueFactor_up',
                    'BSK_3C_VCAM1_down', 'BSK_3C_Vis_down', 'BSK_3C_uPAR_down', 'BSK_4H_Eotaxin3_down',
                    'BSK_4H_MCP1_down', 'BSK_4H_Pselectin_down', 'BSK_4H_Pselectin_up', 'BSK_4H_SRB_down',
                    'BSK_4H_VCAM1_down', 'BSK_4H_VEGFRII_down', 'BSK_4H_uPAR_down', 'BSK_4H_uPAR_up',
                    'BSK_BE3C_HLADR_down', 'BSK_BE3C_IL1a_down', 'BSK_BE3C_IP10_down', 'BSK_BE3C_MIG_down',
                    'BSK_BE3C_MMP1_down', 'BSK_BE3C_MMP1_up', 'BSK_BE3C_PAI1_down', 'BSK_BE3C_SRB_down',
                    'BSK_BE3C_TGFb1_down', 'BSK_BE3C_tPA_down', 'BSK_BE3C_uPAR_down', 'BSK_BE3C_uPAR_up',
                    'BSK_BE3C_uPA_down', 'BSK_CASM3C_HLADR_down', 'BSK_CASM3C_IL6_down', 'BSK_CASM3C_IL6_up',
                    'BSK_CASM3C_IL8_down', 'BSK_CASM3C_LDLR_down', 'BSK_CASM3C_LDLR_up', 'BSK_CASM3C_MCP1_down',
                    'BSK_CASM3C_MCP1_up', 'BSK_CASM3C_MCSF_down', 'BSK_CASM3C_MCSF_up', 'BSK_CASM3C_MIG_down',
                    'BSK_CASM3C_Proliferation_down', 'BSK_CASM3C_Proliferation_up', 'BSK_CASM3C_SAA_down',
                    'BSK_CASM3C_SAA_up', 'BSK_CASM3C_SRB_down', 'BSK_CASM3C_Thrombomodulin_down',
                    'BSK_CASM3C_Thrombomodulin_up', 'BSK_CASM3C_TissueFactor_down', 'BSK_CASM3C_VCAM1_down',
                    'BSK_CASM3C_VCAM1_up', 'BSK_CASM3C_uPAR_down', 'BSK_CASM3C_uPAR_up', 'BSK_KF3CT_ICAM1_down',
                    'BSK_KF3CT_IL1a_down', 'BSK_KF3CT_IP10_down', 'BSK_KF3CT_IP10_up', 'BSK_KF3CT_MCP1_down',
                    'BSK_KF3CT_MCP1_up', 'BSK_KF3CT_MMP9_down', 'BSK_KF3CT_SRB_down', 'BSK_KF3CT_TGFb1_down',
                    'BSK_KF3CT_TIMP2_down', 'BSK_KF3CT_uPA_down', 'BSK_LPS_CD40_down', 'BSK_LPS_Eselectin_down',
                    'BSK_LPS_Eselectin_up', 'BSK_LPS_IL1a_down', 'BSK_LPS_IL1a_up', 'BSK_LPS_IL8_down',
                    'BSK_LPS_IL8_up', 'BSK_LPS_MCP1_down', 'BSK_LPS_MCSF_down', 'BSK_LPS_PGE2_down', 'BSK_LPS_PGE2_up',
                    'BSK_LPS_SRB_down', 'BSK_LPS_TNFa_down', 'BSK_LPS_TNFa_up', 'BSK_LPS_TissueFactor_down',
                    'BSK_LPS_TissueFactor_up', 'BSK_LPS_VCAM1_down', 'BSK_SAg_CD38_down', 'BSK_SAg_CD40_down',
                    'BSK_SAg_CD69_down', 'BSK_SAg_Eselectin_down', 'BSK_SAg_Eselectin_up', 'BSK_SAg_IL8_down',
                    'BSK_SAg_IL8_up', 'BSK_SAg_MCP1_down', 'BSK_SAg_MIG_down', 'BSK_SAg_PBMCCytotoxicity_down',
                    'BSK_SAg_PBMCCytotoxicity_up', 'BSK_SAg_Proliferation_down', 'BSK_SAg_SRB_down',
                    'BSK_hDFCGF_CollagenIII_down', 'BSK_hDFCGF_EGFR_down', 'BSK_hDFCGF_EGFR_up', 'BSK_hDFCGF_IL8_down',
                    'BSK_hDFCGF_IP10_down', 'BSK_hDFCGF_MCSF_down', 'BSK_hDFCGF_MIG_down', 'BSK_hDFCGF_MMP1_down',
                    'BSK_hDFCGF_MMP1_up', 'BSK_hDFCGF_PAI1_down', 'BSK_hDFCGF_Proliferation_down',
                    'BSK_hDFCGF_SRB_down', 'BSK_hDFCGF_TIMP1_down', 'BSK_hDFCGF_VCAM1_down', 'CEETOX_H295R_11DCORT_dn',
                    'CEETOX_H295R_ANDR_dn', 'CEETOX_H295R_CORTISOL_dn', 'CEETOX_H295R_DOC_dn', 'CEETOX_H295R_DOC_up',
                    'CEETOX_H295R_ESTRADIOL_dn', 'CEETOX_H295R_ESTRADIOL_up', 'CEETOX_H295R_ESTRONE_dn',
                    'CEETOX_H295R_ESTRONE_up', 'CEETOX_H295R_OHPREG_up', 'CEETOX_H295R_OHPROG_dn',
                    'CEETOX_H295R_OHPROG_up', 'CEETOX_H295R_PROG_up', 'CEETOX_H295R_TESTO_dn', 'CLD_ABCB1_48hr',
                    'CLD_ABCG2_48hr', 'CLD_CYP1A1_24hr', 'CLD_CYP1A1_48hr', 'CLD_CYP1A1_6hr', 'CLD_CYP1A2_24hr',
                    'CLD_CYP1A2_48hr', 'CLD_CYP1A2_6hr', 'CLD_CYP2B6_24hr', 'CLD_CYP2B6_48hr', 'CLD_CYP2B6_6hr',
                    'CLD_CYP3A4_24hr', 'CLD_CYP3A4_48hr', 'CLD_CYP3A4_6hr', 'CLD_GSTA2_48hr', 'CLD_SULT2A_24hr',
                    'CLD_SULT2A_48hr', 'CLD_UGT1A1_24hr', 'CLD_UGT1A1_48hr', 'NCCT_HEK293T_CellTiterGLO',
                    'NCCT_QuantiLum_inhib_2_dn', 'NCCT_QuantiLum_inhib_dn', 'NCCT_TPO_AUR_dn', 'NCCT_TPO_GUA_dn',
                    'NHEERL_ZF_144hpf_TERATOSCORE_up', 'NVS_ADME_hCYP19A1', 'NVS_ADME_hCYP1A1', 'NVS_ADME_hCYP1A2',
                    'NVS_ADME_hCYP2A6', 'NVS_ADME_hCYP2B6', 'NVS_ADME_hCYP2C19', 'NVS_ADME_hCYP2C9', 'NVS_ADME_hCYP2D6',
                    'NVS_ADME_hCYP3A4', 'NVS_ADME_hCYP4F12', 'NVS_ADME_rCYP2C12', 'NVS_ENZ_hAChE', 'NVS_ENZ_hAMPKa1',
                    'NVS_ENZ_hAurA', 'NVS_ENZ_hBACE', 'NVS_ENZ_hCASP5', 'NVS_ENZ_hCK1D', 'NVS_ENZ_hDUSP3',
                    'NVS_ENZ_hES', 'NVS_ENZ_hElastase', 'NVS_ENZ_hFGFR1', 'NVS_ENZ_hGSK3b', 'NVS_ENZ_hMMP1',
                    'NVS_ENZ_hMMP13', 'NVS_ENZ_hMMP2', 'NVS_ENZ_hMMP3', 'NVS_ENZ_hMMP7', 'NVS_ENZ_hMMP9',
                    'NVS_ENZ_hPDE10', 'NVS_ENZ_hPDE4A1', 'NVS_ENZ_hPDE5', 'NVS_ENZ_hPI3Ka', 'NVS_ENZ_hPTEN',
                    'NVS_ENZ_hPTPN11', 'NVS_ENZ_hPTPN12', 'NVS_ENZ_hPTPN13', 'NVS_ENZ_hPTPN9', 'NVS_ENZ_hPTPRC',
                    'NVS_ENZ_hSIRT1', 'NVS_ENZ_hSIRT2', 'NVS_ENZ_hTrkA', 'NVS_ENZ_hVEGFR2', 'NVS_ENZ_oCOX1',
                    'NVS_ENZ_oCOX2', 'NVS_ENZ_rAChE', 'NVS_ENZ_rCNOS', 'NVS_ENZ_rMAOAC', 'NVS_ENZ_rMAOAP',
                    'NVS_ENZ_rMAOBC', 'NVS_ENZ_rMAOBP', 'NVS_ENZ_rabI2C', 'NVS_GPCR_bAdoR_NonSelective',
                    'NVS_GPCR_bDR_NonSelective', 'NVS_GPCR_g5HT4', 'NVS_GPCR_gH2', 'NVS_GPCR_gLTB4', 'NVS_GPCR_gLTD4',
                    'NVS_GPCR_gMPeripheral_NonSelective', 'NVS_GPCR_gOpiateK', 'NVS_GPCR_h5HT2A', 'NVS_GPCR_h5HT5A',
                    'NVS_GPCR_h5HT6', 'NVS_GPCR_h5HT7', 'NVS_GPCR_hAT1', 'NVS_GPCR_hAdoRA1', 'NVS_GPCR_hAdoRA2a',
                    'NVS_GPCR_hAdra2A', 'NVS_GPCR_hAdra2C', 'NVS_GPCR_hAdrb1', 'NVS_GPCR_hAdrb2', 'NVS_GPCR_hAdrb3',
                    'NVS_GPCR_hDRD1', 'NVS_GPCR_hDRD2s', 'NVS_GPCR_hDRD4.4', 'NVS_GPCR_hH1', 'NVS_GPCR_hLTB4_BLT1',
                    'NVS_GPCR_hM1', 'NVS_GPCR_hM2', 'NVS_GPCR_hM3', 'NVS_GPCR_hM4', 'NVS_GPCR_hNK2',
                    'NVS_GPCR_hOpiate_D1', 'NVS_GPCR_hOpiate_mu', 'NVS_GPCR_hTXA2', 'NVS_GPCR_p5HT2C',
                    'NVS_GPCR_r5HT1_NonSelective', 'NVS_GPCR_r5HT_NonSelective', 'NVS_GPCR_rAdra1B',
                    'NVS_GPCR_rAdra1_NonSelective', 'NVS_GPCR_rAdra2_NonSelective', 'NVS_GPCR_rAdrb_NonSelective',
                    'NVS_GPCR_rNK1', 'NVS_GPCR_rNK3', 'NVS_GPCR_rOpiate_NonSelective',
                    'NVS_GPCR_rOpiate_NonSelectiveNa', 'NVS_GPCR_rSST', 'NVS_GPCR_rTRH', 'NVS_GPCR_rV1',
                    'NVS_GPCR_rabPAF', 'NVS_GPCR_rmAdra2B', 'NVS_IC_hKhERGCh', 'NVS_IC_rCaBTZCHL', 'NVS_IC_rCaDHPRCh_L',
                    'NVS_IC_rNaCh_site2', 'NVS_LGIC_bGABARa1', 'NVS_LGIC_h5HT3', 'NVS_LGIC_hNNR_NBungSens',
                    'NVS_LGIC_rGABAR_NonSelective', 'NVS_LGIC_rNNR_BungSens', 'NVS_MP_hPBR', 'NVS_MP_rPBR',
                    'NVS_NR_bER', 'NVS_NR_bPR', 'NVS_NR_cAR', 'NVS_NR_hAR', 'NVS_NR_hCAR_Antagonist', 'NVS_NR_hER',
                    'NVS_NR_hFXR_Agonist', 'NVS_NR_hFXR_Antagonist', 'NVS_NR_hGR', 'NVS_NR_hPPARa', 'NVS_NR_hPPARg',
                    'NVS_NR_hPR', 'NVS_NR_hPXR', 'NVS_NR_hRAR_Antagonist', 'NVS_NR_hRARa_Agonist',
                    'NVS_NR_hTRa_Antagonist', 'NVS_NR_mERa', 'NVS_NR_rAR', 'NVS_NR_rMR', 'NVS_OR_gSIGMA_NonSelective',
                    'NVS_TR_gDAT', 'NVS_TR_hAdoT', 'NVS_TR_hDAT', 'NVS_TR_hNET', 'NVS_TR_hSERT', 'NVS_TR_rNET',
                    'NVS_TR_rSERT', 'NVS_TR_rVMAT2', 'OT_AR_ARELUC_AG_1440', 'OT_AR_ARSRC1_0480', 'OT_AR_ARSRC1_0960',
                    'OT_ER_ERaERa_0480', 'OT_ER_ERaERa_1440', 'OT_ER_ERaERb_0480', 'OT_ER_ERaERb_1440',
                    'OT_ER_ERbERb_0480', 'OT_ER_ERbERb_1440', 'OT_ERa_EREGFP_0120', 'OT_ERa_EREGFP_0480',
                    'OT_FXR_FXRSRC1_0480', 'OT_FXR_FXRSRC1_1440', 'OT_NURR1_NURR1RXRa_0480', 'OT_NURR1_NURR1RXRa_1440',
                    'TOX21_ARE_BLA_Agonist_ch1', 'TOX21_ARE_BLA_Agonist_ch2', 'TOX21_ARE_BLA_agonist_ratio',
                    'TOX21_ARE_BLA_agonist_viability', 'TOX21_AR_BLA_Agonist_ch1', 'TOX21_AR_BLA_Agonist_ch2',
                    'TOX21_AR_BLA_Agonist_ratio', 'TOX21_AR_BLA_Antagonist_ch1', 'TOX21_AR_BLA_Antagonist_ch2',
                    'TOX21_AR_BLA_Antagonist_ratio', 'TOX21_AR_BLA_Antagonist_viability', 'TOX21_AR_LUC_MDAKB2_Agonist',
                    'TOX21_AR_LUC_MDAKB2_Antagonist', 'TOX21_AR_LUC_MDAKB2_Antagonist2', 'TOX21_AhR_LUC_Agonist',
                    'TOX21_Aromatase_Inhibition', 'TOX21_AutoFluor_HEK293_Cell_blue',
                    'TOX21_AutoFluor_HEK293_Media_blue', 'TOX21_AutoFluor_HEPG2_Cell_blue',
                    'TOX21_AutoFluor_HEPG2_Cell_green', 'TOX21_AutoFluor_HEPG2_Media_blue',
                    'TOX21_AutoFluor_HEPG2_Media_green', 'TOX21_ELG1_LUC_Agonist', 'TOX21_ERa_BLA_Agonist_ch1',
                    'TOX21_ERa_BLA_Agonist_ch2', 'TOX21_ERa_BLA_Agonist_ratio', 'TOX21_ERa_BLA_Antagonist_ch1',
                    'TOX21_ERa_BLA_Antagonist_ch2', 'TOX21_ERa_BLA_Antagonist_ratio',
                    'TOX21_ERa_BLA_Antagonist_viability', 'TOX21_ERa_LUC_BG1_Agonist', 'TOX21_ERa_LUC_BG1_Antagonist',
                    'TOX21_ESRE_BLA_ch1', 'TOX21_ESRE_BLA_ch2', 'TOX21_ESRE_BLA_ratio', 'TOX21_ESRE_BLA_viability',
                    'TOX21_FXR_BLA_Antagonist_ch1', 'TOX21_FXR_BLA_Antagonist_ch2', 'TOX21_FXR_BLA_agonist_ch2',
                    'TOX21_FXR_BLA_agonist_ratio', 'TOX21_FXR_BLA_antagonist_ratio',
                    'TOX21_FXR_BLA_antagonist_viability', 'TOX21_GR_BLA_Agonist_ch1', 'TOX21_GR_BLA_Agonist_ch2',
                    'TOX21_GR_BLA_Agonist_ratio', 'TOX21_GR_BLA_Antagonist_ch2', 'TOX21_GR_BLA_Antagonist_ratio',
                    'TOX21_GR_BLA_Antagonist_viability', 'TOX21_HSE_BLA_agonist_ch1', 'TOX21_HSE_BLA_agonist_ch2',
                    'TOX21_HSE_BLA_agonist_ratio', 'TOX21_HSE_BLA_agonist_viability', 'TOX21_MMP_ratio_down',
                    'TOX21_MMP_ratio_up', 'TOX21_MMP_viability', 'TOX21_NFkB_BLA_agonist_ch1',
                    'TOX21_NFkB_BLA_agonist_ch2', 'TOX21_NFkB_BLA_agonist_ratio', 'TOX21_NFkB_BLA_agonist_viability',
                    'TOX21_PPARd_BLA_Agonist_viability', 'TOX21_PPARd_BLA_Antagonist_ch1',
                    'TOX21_PPARd_BLA_agonist_ch1', 'TOX21_PPARd_BLA_agonist_ch2', 'TOX21_PPARd_BLA_agonist_ratio',
                    'TOX21_PPARd_BLA_antagonist_ratio', 'TOX21_PPARd_BLA_antagonist_viability',
                    'TOX21_PPARg_BLA_Agonist_ch1', 'TOX21_PPARg_BLA_Agonist_ch2', 'TOX21_PPARg_BLA_Agonist_ratio',
                    'TOX21_PPARg_BLA_Antagonist_ch1', 'TOX21_PPARg_BLA_antagonist_ratio',
                    'TOX21_PPARg_BLA_antagonist_viability', 'TOX21_TR_LUC_GH3_Agonist', 'TOX21_TR_LUC_GH3_Antagonist',
                    'TOX21_VDR_BLA_Agonist_viability', 'TOX21_VDR_BLA_Antagonist_ch1', 'TOX21_VDR_BLA_agonist_ch2',
                    'TOX21_VDR_BLA_agonist_ratio', 'TOX21_VDR_BLA_antagonist_ratio',
                    'TOX21_VDR_BLA_antagonist_viability', 'TOX21_p53_BLA_p1_ch1', 'TOX21_p53_BLA_p1_ch2',
                    'TOX21_p53_BLA_p1_ratio', 'TOX21_p53_BLA_p1_viability', 'TOX21_p53_BLA_p2_ch1',
                    'TOX21_p53_BLA_p2_ch2', 'TOX21_p53_BLA_p2_ratio', 'TOX21_p53_BLA_p2_viability',
                    'TOX21_p53_BLA_p3_ch1', 'TOX21_p53_BLA_p3_ch2', 'TOX21_p53_BLA_p3_ratio',
                    'TOX21_p53_BLA_p3_viability', 'TOX21_p53_BLA_p4_ch1', 'TOX21_p53_BLA_p4_ch2',
                    'TOX21_p53_BLA_p4_ratio', 'TOX21_p53_BLA_p4_viability', 'TOX21_p53_BLA_p5_ch1',
                    'TOX21_p53_BLA_p5_ch2', 'TOX21_p53_BLA_p5_ratio', 'TOX21_p53_BLA_p5_viability',
                    'Tanguay_ZF_120hpf_AXIS_up', 'Tanguay_ZF_120hpf_ActivityScore', 'Tanguay_ZF_120hpf_BRAI_up',
                    'Tanguay_ZF_120hpf_CFIN_up', 'Tanguay_ZF_120hpf_CIRC_up', 'Tanguay_ZF_120hpf_EYE_up',
                    'Tanguay_ZF_120hpf_JAW_up', 'Tanguay_ZF_120hpf_MORT_up', 'Tanguay_ZF_120hpf_OTIC_up',
                    'Tanguay_ZF_120hpf_PE_up', 'Tanguay_ZF_120hpf_PFIN_up', 'Tanguay_ZF_120hpf_PIG_up',
                    'Tanguay_ZF_120hpf_SNOU_up', 'Tanguay_ZF_120hpf_SOMI_up', 'Tanguay_ZF_120hpf_SWIM_up',
                    'Tanguay_ZF_120hpf_TRUN_up', 'Tanguay_ZF_120hpf_TR_up', 'Tanguay_ZF_120hpf_YSE_up'],
        'ESOL': ['logSolubility'],
        'FreeSolv': ['freesolv'],
        'LIPO': ['lipo']
    }
    # Prepare header for results which includes task-property labels
    headers = ["drug_smiles"]
    total_properties = 1
    for task in task_list:
        for prop in task_properties[task]:
            headers.append(f"{task}-{prop}")
            total_properties += 1

    property_values = [headers]  # Start with headers

    # Temporary storage for drug properties across all tasks, initialize with Nones
    all_drugs_properties = [[np.nan] * total_properties for _ in drug_smiles]
    for i in range(len(drug_smiles)):
        all_drugs_properties[i][0] = drug_smiles[i]

    cur_path = os.path.dirname(__file__)
    for task in task_list:
        config_path = f"{cur_path}/configs/{task}.yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        config['cuda_name'] = device
        config['batch_size'] = batch_size

        model_type = Model_CLS if task in cls_tasks else Model_REG
        model_file_path = f"{cur_path}/pretrained_models/{task}_best_model.model"
        best_model = model_type(config)
        best_model.load_state_dict(torch.load(model_file_path, map_location=torch.device(device)), strict=False)
        best_model.to(device)

        testset, valid_indices = load_data(drug_smiles, task)
        testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)

        for data in testset_loader:
            props = predicting(best_model, device, [data], "test", config)
            if task in cls_tasks:
                props = torch.sigmoid(torch.tensor(props)).numpy()  # Classification task
            props = props.reshape(-1, len(task_properties[task]))

            for idx, prop in zip(valid_indices, props):
                offset = headers.index(f"{task}-{task_properties[task][0]}")  # Find where to place properties in row
                for i, p in enumerate(prop):
                    all_drugs_properties[idx][offset + i] = p

    # Replace rows that are entirely None with a single None
    property_values.extend([props for props in all_drugs_properties])
    output = {}
    output['result'] = task_descriptions[task]
    output['result_values'] = []
    for i in range(1, len(property_values)):
        result = {}
        result['smile'] = property_values[i][0]
        result['value'] = property_values[i][1]
        output['result_values'].append(result)
    # print(property_values)s
    return output

# 查找子属性所属的任务，即数据集，若没有该子属性则返回None
def get_task(property):
    task_properties = {
        'BACE': ['Class'],
        'BBBP': ['p_np'],
        'ClinTox': ['FDA_APPROVED', 'CT_TOX'],
        'Tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
                  'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
        'ESOL': ['logSolubility'],
        'FreeSolv': ['freesolv'],
        'LIPO': ['lipo']
    }
    for task, properties in task_properties.items():
        if property in properties:
            return task
    return None

if __name__ == "__main__":
    device = 'cuda:5'
    batch_size = 32


    smile= ['C1=CC=CC=C1C', 'COC1=C(C=C2C(=C1)CCN=C2C3=CC(=C(C=C3)Cl)Cl)Cl']
    task=  ""
    
    output = drug_property_prediction(smile, 'Class')
    print(output)
    
    # drug_property_prediction返回参数示例
    # [['BACE-Class', 'ClinTox-FDA_APPROVED', 'ClinTox-CT_TOX', 'FreeSolv-freesolv'],
    # [0.01808778, 0.9999472, 0.00014919841, -5.666918],
    # [None, None, None, None],
    # [0.0012767817, 0.9982792, 0.003367875, -5.9506307]]
