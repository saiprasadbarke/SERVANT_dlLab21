from json import load
from torchtext.vocab import Vocab, build_vocab_from_iterator
from settings import special_symbols, UNK_IDX


class ParseWeightsAndEquationsJson:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.weights, self.tokenized_equations = self.parse_equations_weights_json()

    def parse_equations_weights_json(self):
        json_file = open(self.root_dir)
        equations_weights_data = load(json_file)
        equations = list(equations_weights_data.keys())
        weights = list(equations_weights_data.values())
        tokenized_equations = [
            ParseWeightsAndEquationsJson.tokenize_equation(equation)
            for equation in equations
        ]
        return weights, tokenized_equations

    def get_equations_vocab_transform(self) -> Vocab:
        vocab_transform = build_vocab_from_iterator(
            self.tokenized_equations,
            min_freq=1,
            specials=special_symbols,
            special_first=True,
        )
        vocab_transform.set_default_index(UNK_IDX)
        return vocab_transform

    @staticmethod
    def tokenize_equation(equation):
        "Tokenizes an equation and returns a list"
        token_list = [token for token in equation]
        return token_list


if __name__ == "__main__":
    root_dir = "./network_wts_eqs_dataset/ntwrk_wts_eqs_1000.json"
    data = ParseWeightsAndEquationsJson(root_dir)
    vocab_transform = data.get_equations_vocab_transform()
    print(f"Vocab_dict string to index: {vocab_transform.get_stoi()}")
    print(f"Vocab_dict index to string: {vocab_transform.get_itos()}")
    print(f"Length of vocab: {vocab_transform.__len__()}")
    print(
        f"Numericalization of {data.tokenized_equations[0]} is {vocab_transform(data.tokenized_equations[0])}"
    )
