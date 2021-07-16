import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        # call constructor from superclass
        super(MLPEncoder, self).__init__()
        # define network layers
        self.input_size = input_size

        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(num_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(num_features=hidden_size)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=hidden_size * 2)
        self.batch_norm3 = nn.BatchNorm1d(num_features=hidden_size * 2)
        self.fc4 = nn.Linear(in_features=hidden_size * 2, out_features=output_size)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(output_size)

    def forward(self, x):
        # define forward pass
        output = self.fc1(x)
        output = self.batch_norm1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.batch_norm2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.batch_norm3(output)
        output = self.relu(output)
        output = self.fc4(output)
        output = self.layer_norm1(output)
        return output
