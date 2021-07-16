from parse_equations_weights import ParseEquationsWeights
from torch.utils.data import Dataset, DataLoader


class EquationsWeightsDataset(Dataset):
    """
    This class extends the Dataset implementation from torch.utils.data.Dataset. The 3 methods below have to be overridden.
    """

    def __init__(self, X_weights, y_equations):
        self.y_equations = y_equations
        self.X_weights = X_weights

    def __len__(self):
        return len(self.y_equations)

    def __getitem__(self, idx):
        x_value = self.X_weights[idx]
        y_value = self.y_equations[idx]
        return x_value, y_value


if __name__ == "__main__":

    root_dir = "./network_wts_eqs_dataset/ntwrk_wts_eqs_1000.json"
    equation_weight_dataset = EquationsWeightsDataset(root_dir)
    data_loader = DataLoader(equation_weight_dataset, batch_size=1, shuffle=True)
    for idx, xy_values in enumerate(data_loader):
        print(f"XY at position {idx} is {xy_values}")
