import torch.nn as nn

class Simple_MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        # call constructor from superclass
        super(Simple_MLP, self).__init__()
        # define network layers
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 7)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(7, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # define forward pass
        output = self.fc1(x)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        output = self.relu3(output)
        output = self.fc4(output)
        return output