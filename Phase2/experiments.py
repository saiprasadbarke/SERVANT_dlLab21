# Local
from inference import test_model
from train_network import create_train_val_test_dataloaders, create_model
from settings import root_dir_50k_16, root_dir_100k, root_dir_1k, DEVICE, PAD_IDX

# External
from torch import load
from train_eval import eval_model
import torch.nn as nn
import sys

_, _, test_dataloader = create_train_val_test_dataloaders(root_dir_100k, 1)
model = create_model(6, 8, 1024)
run_id = "bigrun_3"
model.load_state_dict(load(f"./runs/{run_id}/model_{run_id}.pth"))
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
test_loss = eval_model(test_dataloader, model, criterion, DEVICE)
sys.stdout = open(f"./runs/{run_id}/test_results_{run_id}.txt", "w")
print(f"Final test loss : {test_loss}")
test_model(test_dataloader, model, "beam")
