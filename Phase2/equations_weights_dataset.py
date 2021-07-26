from torch.utils.data import Dataset


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
