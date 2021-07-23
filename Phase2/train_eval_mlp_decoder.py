import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from symbolic_regression_transformer import SymbolicRegressionTransformer
from masking import create_mask


def train_model(
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    epochs,
    model: SymbolicRegressionTransformer,
    optimizer,
    scheduler,
    criterion,
    device,
):
    train_losses = []
    validation_losses = []
    best_val_loss = np.inf
    epochs_no_improve = 0
    n_epochs_stop = 10
    # train-validation loop
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        batch_losses = []
        training_loss = 0.0
        # training loop
        for _idx, data in enumerate(train_dataloader):
            source_weights, target_embedded_equation = data
            source_weights, target_embedded_equation = (
                source_weights.to(device),
                target_embedded_equation.to(device),
            )
            target_embedded_equation_input = target_embedded_equation[:-1, :]
            tgt_mask, tgt_padding_mask = create_mask(target_embedded_equation_input)
            optimizer.zero_grad()
            model.train()
            mlp_transformerdecoder_logits = model(
                source_weights,
                target_embedded_equation_input,
                tgt_mask,
                tgt_padding_mask,
            )
            target_embedded_equation_out = target_embedded_equation[1:, :].reshape(-1)
            mlp_transformerdecoder_logits_out = mlp_transformerdecoder_logits.reshape(
                -1, mlp_transformerdecoder_logits.shape[-1]
            )
            loss = criterion(
                mlp_transformerdecoder_logits_out,
                target_embedded_equation_out,
            )
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        training_loss = np.mean(batch_losses)
        train_losses.append(training_loss)
        scheduler.step()

        # validation loop
        with torch.no_grad():
            val_losses = []
            validation_loss = 0.0
            for _idx, data in enumerate(validation_dataloader):
                source_weights, target_embedded_equation = data
                source_weights, target_embedded_equation = (
                    source_weights.to(device),
                    target_embedded_equation.to(device),
                )
                target_embedded_equation_input = target_embedded_equation[:-1, :]
                tgt_mask, tgt_padding_mask = create_mask(target_embedded_equation_input)
                model.eval()
                mlp_transformerdecoder_logits = model(
                    source_weights,
                    target_embedded_equation_input,
                    tgt_mask,
                    tgt_padding_mask,
                )
                target_embedded_equation_out = target_embedded_equation[1:, :].reshape(
                    -1
                )
                mlp_transformerdecoder_logits_out = (
                    mlp_transformerdecoder_logits.reshape(
                        -1, mlp_transformerdecoder_logits.shape[-1]
                    )
                )
                loss = criterion(
                    mlp_transformerdecoder_logits_out,
                    target_embedded_equation_out,
                )
                val_losses.append(loss.item())
            validation_loss = np.mean(val_losses)
            validation_losses.append(validation_loss)
            print(
                f"[{epoch}] Training loss: {training_loss:.7f}\t Validation loss: {validation_loss:.7f}"
            )
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == n_epochs_stop:
                    print(f"Early stopping on epoch number {epoch}!")
                    break

        # print(f"\t Label value: {labels.float().item()}\t Predicted Output: {outputs.float().item()}")
    # torch.save(model.state_dict(), MODEL_PATH)
    return model, train_losses, validation_losses
