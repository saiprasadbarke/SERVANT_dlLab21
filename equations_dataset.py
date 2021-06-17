from json import load
from torch.utils.data import Dataset, DataLoader
from torch import set_printoptions


class EquationsDataset(Dataset):
    """
    This class extends the Dataset implementation from torch.utils.data.Dataset. The 3 methods below have to be overridden.
    """

    def __init__(self, dataset_file_path, x_transform=None, y_transform=None):
        self.dataset_file_path = dataset_file_path
        (
            self.equation_name,
            self.x_values,
            self.y_values,
        ) = EquationsDataset.parse_equation_json(self.dataset_file_path)
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.y_values)

    def __getitem__(self, idx):
        x_value = self.x_values[idx]
        y_value = self.y_values[idx]
        if self.x_transform:
            x_value = self.x_transform(x_value)
        if self.y_transform:
            y_value = self.y_transform(y_value)
        return x_value, y_value

    @staticmethod
    def parse_equation_json(dataset_file_path):
        with open(dataset_file_path) as json_file:
            data = load(json_file)
            equation_name = data["equation"]
            xy_values_dict = data["xy_values"]
            x_values = list(xy_values_dict.keys())
            x_values = [float(x_value) for x_value in x_values]
            y_values = list(xy_values_dict.values())
            assert len(x_values) == len(y_values)
        return equation_name, x_values, y_values


if __name__ == "__main__":

    root_dir = "./datasets"
    equation_data = EquationsDataset(dataset_file_path=f"{root_dir}/Equation1.json")
    data_loader = DataLoader(equation_data, batch_size=1, shuffle=True)
    for idx, xy_values in enumerate(data_loader):
        print(f"XY at position {idx} is {xy_values}")
