from inference import test_model
from train_network import create_train_val_test_dataloaders, create_model
from settings import root_dir
from torch import load

_, _, test_dataloader = create_train_val_test_dataloaders(root_dir, 1)
model = create_model()
model.load_state_dict(load("./models/bigrun_2_model.pth"))
test_model(test_dataloader, model, "beam")
