# Local
from inference import test_model
from train_network import create_train_val_test_dataloaders, create_model
from settings import (
    DEVICE,
    PAD_IDX,
    root_dir_50k_16,
    root_dir_1k_test,
    root_dir_100k_non_uniform,
    root_dir_10k_uniform,
    root_dir_14_16_18_uniform,
)
from nearness_score import compute_nearness_signs

# External
from torch import load
from train_eval import eval_model
import torch.nn as nn
import sys
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--run_id",
        type=str,
        default="bigrun_4_16seq",
        help="Specify a run_id for the training run",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="50k_16",
        help="Specify a dataset to test the network on",
    )
    parser.add_argument(
        "--search_type",
        type=str,
        dest="search_type",
        default="beam",
        help="Specify either 'beam' for beam search or 'greedy' for greedy search",
    )
    args = parser.parse_args()
    # Load the required dataset from path
    if args.dataset == "100K_non_uniform":
        dataset = root_dir_100k_non_uniform
    elif args.dataset == "10k_uniform":
        dataset = root_dir_10k_uniform
    elif args.dataset == "50k_16":
        dataset = root_dir_50k_16
    elif args.dataset == "14_16_18_uniform":
        dataset = root_dir_14_16_18_uniform
    elif args.dataset == "test_dataset":
        dataset = root_dir_1k_test

    _, _, test_dataloader = create_train_val_test_dataloaders(dataset, 1, 0.1)
    model = create_model(6, 8, 1024)
    model.load_state_dict(load(f"./runs/{args.run_id}/model_{args.run_id}.pth"))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    test_loss = eval_model(test_dataloader, model, criterion, DEVICE)
    sys.stdout = open(f"./runs/{args.run_id}/test_results_{args.run_id}.txt", "w")
    print(f"Running test with args: {args}")
    print(f"Final test loss : {test_loss}")
    ground_truth_equations, predicted_equations = test_model(
        test_dataloader, model, args.search_type
    )
    nearness_score = compute_nearness_signs(ground_truth_equations, predicted_equations)
    print(f"Nearness score for correct prediction of signs is {nearness_score} %")
