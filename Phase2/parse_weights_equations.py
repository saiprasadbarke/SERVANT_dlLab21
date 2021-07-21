from json import load
from torchtext.vocab import build_vocab_from_iterator, Vocab
from settings import UNK_IDX, special_symbols


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

    @staticmethod
    def tokenize_equation(equation):
        "Tokenizes an equation and returns a list"
        token_list = [token for token in equation]
        return token_list


if __name__ == "__main__":
    root_dir = "./network_wts_eqs_dataset/ntwrk_wts_eqs_1000.json"
    data = ParseWeightsAndEquationsJson(root_dir)
    vocab_transform = data.build_equation_tokens_vocab_object()
    print(f"Vocab_dict string to index: {vocab_transform.get_stoi()()}")
    print(f"Vocab_dict index to string: {vocab_transform.get_itos()}")
    print(
        f"Numericalization of {data.tokenized_equations[0]} is {vocab_transform(data.tokenized_equations[0])}"
    )
