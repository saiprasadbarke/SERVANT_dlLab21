import numpy as np
import json

MAX_LENGTH = 18


class EquationEmbedding:
    def __init__(self, equation):
        self.equation = equation

    def embedding_module(self):
        "Our vocabulary has 15 characters, numbers from 0-9, the 3 operators, a decimal point and the variable x. We will represent each one by a one hot vector with 15 bins. Each equation will number of rows corresponding to the number of characters in the equation token list."
        embedded_equation = []
        token_list = self.split()
        number_of_paddings = MAX_LENGTH - len(token_list)
        embedded_equation.append(self.generate_embedding("SOS"))
        for token in token_list:
            embedded_equation.append(self.generate_embedding(token))
        embedded_equation.append(self.generate_embedding("EOS"))
        for _idx in range(number_of_paddings):
            embedded_equation.append(self.generate_embedding("PAD"))
        return np.asarray(embedded_equation, dtype="float64")

    def split(self):
        return [token for token in self.equation]

    def generate_embedding(self, token):
        embedding_dict = {
            "0": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            "1": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            "2": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
            "3": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            "4": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            "5": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            "6": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            "7": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            "8": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            "9": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "+": np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "-": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "*": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ".": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "x": np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "EOS": np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "SOS": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "PAD": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        }
        return embedding_dict[token]


if __name__ == "__main__":

    dir = "./network_wts_eqs_dataset/Equation1.json"
    json_file = open(dir)
    equation_embedder = EquationEmbedding(equation=json.load(json_file)["equation"])
    array = equation_embedder.embedding_module()
    print(array)
