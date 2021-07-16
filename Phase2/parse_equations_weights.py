from json import load
from equation_embedding import EquationEmbedding
import numpy as np


class ParseEquationsWeights:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def parse_equations_weights_json(self):
        json_file = open(self.root_dir)
        equations_weights_data = load(json_file)
        equations = list(equations_weights_data.keys())
        weights = list(equations_weights_data.values())
        return equations, weights

    def equations_weights_numpy_array(self):
        equations, weights = self.parse_equations_weights_json()
        embedded_equations = []
        for equation in equations:
            equation_embedder = EquationEmbedding(equation=equation)
            equation_embedding = equation_embedder.embedding_module()
            embedded_equations.append(equation_embedding)
        weights_array = [
            np.asarray(weights_single_eqn) for weights_single_eqn in weights
        ]

        return embedded_equations, weights_array


if __name__ == "__main__":
    root_dir = "./network_wts_eqs_dataset/ntwrk_wts_eqs_1000.json"
    lol = ParseEquationsWeights(root_dir)
    embedded_equations, weights_array = lol.equations_weights_numpy_array()
    print("hello")
