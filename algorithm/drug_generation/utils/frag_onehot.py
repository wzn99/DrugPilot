import numpy as np
import torch
import pandas as pd
import pdb
from tqdm import tqdm

# # df = pd.read_csv('/home/nas/hlt/GDSS/data/zinc_sampled_frags.csv')
# df = pd.read_csv('./data/zinc_frags_total_split.csv')
# smiles_list = df['SMILES2'].tolist()
# # pdb.set_trace()
# unique_chars = sorted(list(set(''.join(smiles_list))))
# # unique_chars = ['-', 's', '#', 'r', ')', '(', 'l', 'B', '4', '2', 'I', '1', '3', '+', '=', 'F', 'O', 'o', ']', 'S', 'c', 'C', 'H', 'n', '*', '[', 'N']
# # unique_chars = ['-', 's', '#', 'r', ')', '(', 'l', 'B', '4', '2', 'I', '1', '3', '+', '=', 'F', 'O', 'o', ']', 'S', 'c', 'C', 'H', 'n', '*', '[', 'N','?']

# char_to_index = {char: i for i, char in enumerate(unique_chars)}
# index_to_char = {i: char for char, i in char_to_index.items()}

# max_length = max(len(string) for string in smiles_list)

def string_to_onehot(string):
    onehot = np.zeros((len(unique_chars) + 1, max_length + 1), dtype=np.float32)
    for i, char in enumerate(string):
        onehot[char_to_index[char], i] = 1
    onehot[-1, len(string)] = 1
    return onehot

def onehot_to_string(onehot):
    char_indices = np.argmax(onehot[:-1, :], axis=0)
    chars = [index_to_char[idx] for idx in char_indices]
    length = np.argmax(onehot[-1, :])
    string = ''.join(chars)[:length]
    return string

def change_type(feat_array):
    onehot_list = [string_to_onehot(string) for string in feat_array.flatten()]
    padded_onehot_list = [torch.tensor(onehot[:, :max_length + 1]) for onehot in onehot_list]
    tensor_data = np.stack(padded_onehot_list)
    return tensor_data


# string_list = [onehot_to_string(tensor.numpy()) for tensor in tensor_data]
# string_list = [onehot_to_string(tensor) for tensor in tensor_data]
