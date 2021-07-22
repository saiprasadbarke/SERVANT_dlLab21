import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from symbolic_regression_transformer import SymbolicRegressionTransformer


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
    # TODO: compute target mask and target padding mask
    # train-validation loop
    for epoch in range(epochs):
        batch_losses = []
        training_loss = 0.0
        # training loop
        for _idx, data in enumerate(train_dataloader):
            source_weights, target_embedded_equation = data
            source_weights, target_embedded_equation_permuted = (
                source_weights.to(device),
                target_embedded_equation.to(device).permute(1, 0, 2),
            )
            optimizer.zero_grad()
            model.train()
            ml_transformerdecoder_outputs = model(
                source_weights, target_embedded_equation_permuted
            )
            loss = criterion(
                ml_transformerdecoder_outputs.permute(1, 2, 0), target_embedded_equation
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
                source_weights, target_embedded_equation = source_weights.to(
                    device
                ), target_embedded_equation.to(device)
                model.eval()
                ml_transformerdecoder_outputs = model(
                    source_weights, target_embedded_equation
                )
                loss = criterion(
                    ml_transformerdecoder_outputs.float(),
                    target_embedded_equation.float(),
                )
                val_losses.append(loss.item())
            validation_loss = np.mean(val_losses)
            validation_losses.append(validation_loss)

        print(
            f"[{epoch+1}] Training loss: {training_loss:.7f}\t Validation loss: {validation_loss:.7f}"
        )
        # print(f"\t Label value: {labels.float().item()}\t Predicted Output: {outputs.float().item()}")
    # torch.save(model.state_dict(), MODEL_PATH)
    return model.state_dict(), train_losses, validation_losses


def eval_model(
    test_dataloader: DataLoader,
    model: MLPTransformerDecoder,
    criterion,
):
    test_losses = []
    with torch.no_grad():
        for _idx, data in enumerate(test_dataloader):
            inputs, labels = data
            model.eval()
            outputs = model(inputs)
            # print("outputs, ", outputs.shape)
            # rescaled_outputs = inverse_scaler(outputs, method="minmax")
            # print("rescaled_outputs: ",rescaled_outputs.shape)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
        test_loss = np.mean(test_losses)
        print(f"Final test loss: {test_loss:.4f}")
    return test_losses
