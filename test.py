from parse_equations_weights import ParseEquationsWeights
from equation_embedding import EquationEmbedding

if __name__ == "__main__":
    root_dir = "./network_wts_eqs_dataset/ntwrk_wts_eqs_1000.json"
    parser = ParseEquationsWeights(root_dir)
    equations, weights = parser.parse_equations_weights_json()
    embedded_equations = []
    for equation in equations:
        equation_embedder = EquationEmbedding(equation=equation)
        equation_embedding = equation_embedder.embedding_module()
        embedded_equations.append(equation_embedding)
