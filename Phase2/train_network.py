# Local
from torch.functional import norm
from equations_weights_dataset import EquationsWeightsDataset
from parse_weights_equations import ParseWeightsAndEquationsJson
from mlp_encoder import MLPEncoder
from symbolic_regression_transformer import SymbolicRegressionTransformer
from generator import Generator
from token_embedding import TokenEmbedding
from positional_encoding import PositionalEncoding
from train_eval import train_model, eval_model
from data_helpers import collate_fn
from settings import DEVICE, PAD_IDX, root_dir, root_dir_test
from vocab_transform import VOCAB_TRANSFORM
from inference import test_model

# External
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import TransformerDecoderLayer, TransformerDecoder, LayerNorm
import torch.nn as nn
import torch
from timeit import default_timer as timer
from numpy import inf


def create_train_val_test_dataloaders(root_dir, batch_size):
    data = ParseWeightsAndEquationsJson(root_dir)
    embedded_equations, weights = data.tokenized_equations, data.weights
    (
        X_train_val_weights,
        X_test_weights,
        y_train_val_embedded_equations,
        y_test_embedded_equations,
    ) = train_test_split(
        weights,
        embedded_equations,
        test_size=0.001,
        random_state=42,
    )
    (
        X_train_weights,
        X_val_weights,
        y_train_embedded_equations,
        y_val_embedded_equations,
    ) = train_test_split(
        X_train_val_weights,
        y_train_val_embedded_equations,
        test_size=0.2,
        random_state=42,
    )
    train_dataset = EquationsWeightsDataset(X_train_weights, y_train_embedded_equations)
    val_dataset = EquationsWeightsDataset(X_val_weights, y_val_embedded_equations)
    test_dataset = EquationsWeightsDataset(X_test_weights, y_test_embedded_equations)

    print(f"Size of Training dataset: {train_dataset.__len__()}")
    print(f"Size of Validation dataset: {val_dataset.__len__()}")
    print(f"Size of Test dataset: {test_dataset.__len__()}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    return train_dataloader, val_dataloader, test_dataloader


def create_model():

    # Encoder parameters
    encoder_input = 128
    encoder_hidden = 256
    encoder_output = embedding_size = 512

    # Decoder parameters
    n_layers_decoder = 6

    # Decoder layer parameters
    n_heads_decoder = 8
    decoder_layer_activation_function = "gelu"
    decoder_layer_feed_forward_dimension = 512
    decoder_layer_dropout = 0.1

    # Other parameters
    vocab_size = 19
    max_sequence_length = 20
    target_positional_encoding_layer_dropout = 0.1
    # Building the model
    encoder = MLPEncoder(
        input_size=encoder_input,
        hidden_size=encoder_hidden,
        output_size=encoder_output,
    )
    decoder_layer = TransformerDecoderLayer(
        d_model=embedding_size,
        nhead=n_heads_decoder,
        activation=decoder_layer_activation_function,
        dim_feedforward=decoder_layer_feed_forward_dimension,
        dropout=decoder_layer_dropout,
    )
    decoder_layer_norm = LayerNorm(normalized_shape=embedding_size)
    decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=n_layers_decoder,
        norm=decoder_layer_norm,
    )

    generator = Generator(
        d_model=embedding_size,
        vocab_size=vocab_size,
    )
    target_token_embedding = TokenEmbedding(
        vocab_size=vocab_size,
        emb_size=embedding_size,
    )
    target_positional_embedding = PositionalEncoding(
        embedding_size=embedding_size,
        dropout=target_positional_encoding_layer_dropout,
        max_len=max_sequence_length,
    )

    mlp_transformer_decoder = SymbolicRegressionTransformer(
        encoder=encoder,
        decoder=decoder,
        target_token_embedding=target_token_embedding,
        target_positional_embedding=target_positional_embedding,
        generator=generator,
    )
    for p in mlp_transformer_decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return mlp_transformer_decoder.to(DEVICE)


def train_network(batch_size: int, n_epochs: int, root_dir_dataset: str, run_id: str):
    # Return all dataloaders
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = create_train_val_test_dataloaders(
        root_dir=root_dir_dataset,
        batch_size=batch_size,
    )
    # Create model
    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-05)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, eta_min=1e-06
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # Early stopping criteria
    best_val_loss = inf
    epochs_no_improve = 0
    n_epochs_stop = 5
    training_losses = []
    validation_losses = []
    # Train the model
    print("Training started...")
    for epoch in range(1, n_epochs + 1):
        start_time = timer()
        training_loss = train_model(
            train_dataloader, model, optimizer, scheduler, criterion, DEVICE
        )
        end_time = timer()
        validation_loss = eval_model(val_dataloader, model, criterion, DEVICE)
        print(
            (
                f"Epoch: {epoch}, Train loss: {training_loss:.3f}, Val loss: {validation_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"
            )
        )
        if validation_loss < best_val_loss:
            torch.save(model.state_dict(), f"./models/{run_id}_model.pth")
            best_val_loss = validation_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == n_epochs_stop:
                print(f"Early stopping on epoch number {epoch}!")
                break
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
    test_loss = eval_model(test_dataloader, model, criterion, DEVICE)
    print(f"Final test loss : {test_loss}")
    return model, training_losses, validation_losses, test_dataloader


if __name__ == "__main__":

    model, training_losses, validation_losses, test_dataloader = train_network(
        batch_size=16, n_epochs=50, root_dir_dataset=root_dir, run_id="bigrun_2"
    )
