# Vocab Transform
from parse_weights_equations import ParseWeightsAndEquationsJson
from settings import root_dir_1k_test

data = ParseWeightsAndEquationsJson(root_dir_1k_test)
VOCAB_TRANSFORM = data.get_equations_vocab_transform()
