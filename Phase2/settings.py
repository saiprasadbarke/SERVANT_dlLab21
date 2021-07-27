# Special Symbols Indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
# Device selection
import torch

DEVICE = torch.device("cuda")
root_dir_50k_16 = "./network_wts_eqs_dataset/equations_to_mlp_weights_50k_16.json"
root_dir_1k_test = "./network_wts_eqs_dataset/equations_to_mlp_weights_1k.json"
root_dir_100k_non_uniform = (
    "./network_wts_eqs_dataset/equations_to_mlp_weights_100k_non_uniform.json"
)
root_dir_10k_uniform = (
    "./network_wts_eqs_dataset/equations_to_mlp_weights_10k_uniform.json"
)
root_dir_14_16_18_uniform = (
    "./network_wts_eqs_dataset/equations_to_mlp_weights_35k_seq_14_16_18_uniform.json"
)
