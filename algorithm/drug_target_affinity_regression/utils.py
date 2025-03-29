import os
import torch
import numpy as np
from math import sqrt
from scipy import stats
# import matplotlib.pyplot as plt
import pdb
from rdkit import Chem, RDConfig
import networkx as nx
from torch_geometric import data as DATA


class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type
        self.count = 0
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)


def load_checkpoint(model_path):
    return torch.load(model_path)


def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))


def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))


def cycle(iterable):
    while True:
        print("end")
        for x in iterable:
            yield x


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


#
# def draw_sort_pred_gt(pred,gt,title):
#     # gt = gt.y.cpu().numpy()
#     # pred = pred.squeeze().cpu().detach().numpy()
#     # zipped = zip(gt,pred)
#     # sort_zipped = sorted(zipped,key=lambda x:(x[0]))
#     # data_gt, data_pred = [list(x) for x in zip(*sort_zipped)]
#     # pdb.set_trace()
#     data_gt, data_pred = zip(*sorted(zip(gt,pred)))
#     plt.figure()
#     plt.scatter( np.arange(len(data_gt)),data_gt, s=0.1, alpha=1, label='gt')
#     plt.scatter( np.arange(len(data_gt)),data_pred, s=0.1, alpha=1, label='pred')
#     plt.legend()
#     plt.savefig(title+".png")
#     plt.close()

def num2english(num, PRECISION=2):
    num = str(round(num, PRECISION)).split('.')[1]

    while len(num) != PRECISION:
        num = num + '0'

    L1 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    word = ""
    for i in str(num):
        # pdb.set_trace()
        word = word + " " + L1[int(i)]

    return word


def preprocess_ColdDTA(SMILES, target_sequence):
    VOCAB_PROTEIN = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                     "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                     "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                     "U": 19, "T": 20, "W": 21,
                     "V": 22, "Y": 23, "X": 24,
                     "Z": 25}

    def seqs2int(target):
        return [VOCAB_PROTEIN[s] for s in target.upper()]

    def one_of_k_encoding(x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def atom_features(atom):
        encoding = one_of_k_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'Unknown'])
        encoding += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(
            atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, 'other'])
        encoding += [atom.GetIsAromatic()]

        try:
            encoding += one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]

        return np.array(encoding)

    def mol_to_graph_without_rm(mol):
        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append(feature / np.sum(feature))

        g = nx.DiGraph()
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]
            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            e[(n1, n2)] = e_t

        if len(e) == 0:
            return features, torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))

        return features, edge_index, edge_attr

    data_list = []
    false_flag = []
    for smiles in SMILES:
        smi = smiles
        sequence = target_sequence
        # affinity = row['affinity']
        # protein_name = row['protein_name']
        mol = Chem.MolFromSmiles(smi)
        if mol == None:
            print("Unable to process: ", smi)
            # return None
            false_flag.append(False)
            continue
        else:
            false_flag.append(True)

        x, edge_index, edge_attr = mol_to_graph_without_rm(mol)

        target = seqs2int(sequence)
        target_len = 1200
        if len(target) < target_len:
            target = np.pad(target, (0, target_len - len(target)))
        else:
            target = target[:target_len]

        data = DATA.Data(
            x=torch.FloatTensor(x),
            smi=smi,
            edge_index=edge_index,
            edge_attr=edge_attr,
            # y=torch.FloatTensor([affinity]),
            target=torch.LongTensor([target]),
            # protein_name=protein_name
        )
        data_list.append(data)

        # if len(delete_list) > 0:
        # df = df.drop(delete_list, axis=0, inplace=False)
        # df.to_csv(data_path, index=False)
        # print('mol_to_graph_without_rm')
    return data_list, false_flag
