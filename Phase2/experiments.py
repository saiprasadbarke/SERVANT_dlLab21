from inference import test_model
from train_network import create_train_val_test_dataloaders, create_model
from settings import root_dir, DEVICE, PAD_IDX
from torch import load
from train_eval import eval_model
import torch.nn as nn

_, _, test_dataloader = create_train_val_test_dataloaders(root_dir, 1)
model = create_model()
model.load_state_dict(load("./models/testrun_2_model.pth"))
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
test_loss = eval_model(test_dataloader, model, criterion, DEVICE)
print(f"Final test loss : {test_loss}")
test_model(test_dataloader, model, "beam")
