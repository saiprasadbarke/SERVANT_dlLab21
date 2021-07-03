from equations_dataset import EquationsDataset
import torch.nn as nn
from json import dump
from torch.utils.data import DataLoader
import torch.optim as optim
from train_eval import train_model, eval_model
from sklearn.model_selection import train_test_split
from os import mkdir, path
from simple_mlp import Simple_MLP


def write_data_to_file(equation_number, equation_name, networks_weights, train_losses, validation_losses):
    root_dir = "./network_wts_eqs_dataset"
    if not path.isdir(root_dir):
        mkdir(root_dir)

    equation_json = dict()
    equation_json["equation"] = equation_name
    equation_json["network_weights"] = networks_weights
    equation_json["train_losses"] = train_losses
    equation_json["validation_losses"] = validation_losses
    with open(f"{root_dir}/Equation{equation_number}.json", "w") as outfile:
        dump(equation_json, outfile, indent=4)

def train_networks_save_weights():
    root_dir = "./datasets"
    for equation_number in range(1, 6):
        equation_data = EquationsDataset(dataset_file_path=f"{root_dir}/Equation{equation_number}.json")
        print(f"generating mlp network weight for : {equation_data.equation_name}")

        train_values_x = train_values_y = test_values_x = test_values_y = None
        train_values_x, train_values_y = equation_data.x_values[:4000], equation_data.y_values[:4000] 
        test_values_x, test_values_y = equation_data.x_values[4000:], equation_data.x_values[4000:] 

        X_train, X_val, y_train, y_val = train_test_split(train_values_x, train_values_y, test_size=0.33, random_state=42)

        train_data = []
        for i in range(len(X_train)):
            train_data.append([X_train[i], y_train[i]])

        val_data = []
        for i in range(len(X_val)):
            val_data.append([X_val[i], y_val[i]])

        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

        model = Simple_MLP(1, 8, 1)
        print(model)

        epochs = 1
        optimizer  = optim.Adam(model.parameters(), lr=1e-05)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-05)
        criterion = nn.MSELoss()

        mlp_state_dict, train_losses, validation_losses = train_model(train_loader, val_loader, epochs, model, optimizer, scheduler, criterion)
        networks_weights = []
        for keys in mlp_state_dict.keys():
            if "bias" not in keys:
                networks_weights.append(mlp_state_dict[keys].flatten().tolist())

        networks_weights = [item for sublist in networks_weights for item in sublist]
        write_data_to_file(equation_number, equation_data.equation_name, networks_weights, train_losses, validation_losses)


if __name__ == "__main__":
    train_networks_save_weights()