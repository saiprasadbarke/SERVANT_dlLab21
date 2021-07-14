from json import load


class ParseEquationsWeightsDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def parse_equations_weights_json(self):
        json_file = open(self.root_dir)
        equations_weights_data = load(json_file)
        equations = list(equations_weights_data.keys())
        weights = list(equations_weights_data.values())
        return equations, weights


if __name__ == "__main__":
    root_dir = "./network_wts_eqs_dataset/ntwrk_wts_eqs_1000.json"
    lol = ParseEquationsWeightsDataset(root_dir)
    lol.parse_equations_weights_json()
