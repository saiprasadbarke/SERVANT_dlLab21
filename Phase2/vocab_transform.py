# Vocab Transform
from parse_weights_equations import ParseWeightsAndEquationsJson

root_dir = "./network_wts_eqs_dataset/ntwrk_wts_eqs_1000.json"
data = ParseWeightsAndEquationsJson(root_dir)
VOCAB_TRANSFORM = data.get_equations_vocab_transform()
