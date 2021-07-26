# Special Symbols Indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
# Device selection
import torch

DEVICE = torch.device("cuda")
root_dir_50k_16 = "./network_wts_eqs_dataset/equations_to_mlp_weights_43k_seq_16.json"
root_dir_1k = "./network_wts_eqs_dataset/ntwrk_wts_eqs_1000.json"
root_dir_100k = "./network_wts_eqs_dataset/equations_to_mlp_weights__87k"
