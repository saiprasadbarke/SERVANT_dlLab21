# Vocab Transform
from parse_weights_equations import ParseWeightsAndEquationsJson
from settings import root_dir

data = ParseWeightsAndEquationsJson(root_dir)
VOCAB_TRANSFORM = data.get_equations_vocab_transform()
