# Local
from equations_weights_dataset import EquationsWeightsDataset
from parse_weights_equations import ParseWeightsAndEquationsJson
from mlp_encoder import MLPEncoder
from symbolic_regression_transformer import SymbolicRegressionTransformer
from train_eval_mlp_decoder import train_model, eval_model
from data_helpers import collate_fn

# External
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import TransformerDecoderLayer, TransformerDecoder
import torch.nn as nn
import torch


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
        test_size=0.2,
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
        test_size=0.25,
        random_state=42,
    )
    train_dataset = EquationsWeightsDataset(X_train_weights, y_train_embedded_equations)
    val_dataset = EquationsWeightsDataset(X_val_weights, y_val_embedded_equations)
    test_dataset = EquationsWeightsDataset(X_test_weights, y_test_embedded_equations)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    return train_dataloader, val_dataloader, test_dataloader


def create_model():
    # Encoder parameters
    encoder_input = 128
    encoder_hidden = 256
    encoder_output = embedding_dimension = 512
    # Decoder parameters
    max_sequence_length = 20
    n_layers_decoder = 3
    # Decoder layer parameters
    n_heads_decoder = 6
    decoder_layer_activation_function = "relu"
    decoder_layer_feed_forward_dimension = 512
    decoder_layer_dropout = 0.3
    # Building the model
    encoder = MLPEncoder(
        input_size=encoder_input,
        hidden_size=encoder_hidden,
        output_size=encoder_output,
    )
    decoder_layer = TransformerDecoderLayer(
        d_model=embedding_dimension,
        nhead=n_heads_decoder,
        activation=decoder_layer_activation_function,
        dim_feedforward=decoder_layer_feed_forward_dimension,
        dropout=decoder_layer_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer=decoder_layer, num_layers=n_layers_decoder
    )
    # TODO: add generator, target embedding layer, target positional encoding layer
    # TODO: compute target mask and target padding mask
    mlp_transformer_decoder = SymbolicRegressionTransformer(
        encoder=encoder, decoder=decoder
    )
    return mlp_transformer_decoder


def train_network():
    batch_size = 10
    device = torch.device("cuda")
    epochs = 1
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = create_train_val_test_dataloaders(
        "./network_wts_eqs_dataset/ntwrk_wts_eqs_1000.json",
        batch_size=batch_size,
    )
    model = create_model().to(device).double()
    optimizer = optim.Adam(model.parameters(), lr=1e-04)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, eta_min=1e-04
    )
    criterion = nn.CrossEntropyLoss()

    mlp_state_dict = train_model(
        train_dataloader,
        val_dataloader,
        epochs,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
    )


if __name__ == "__main__":

    train_network()
