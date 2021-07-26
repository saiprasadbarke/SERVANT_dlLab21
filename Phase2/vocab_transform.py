# Vocab Transform
from parse_weights_equations import ParseWeightsAndEquationsJson
from settings import root_dir_1k, root_dir_50k_16, root_dir_100k

data = ParseWeightsAndEquationsJson(root_dir_100k)
VOCAB_TRANSFORM = data.get_equations_vocab_transform()
