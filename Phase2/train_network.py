# Local
from equations_weights_dataset import EquationsWeightsDataset
from parse_equations_weights import ParseEquationsWeights
from mlp_encoder import MLPEncoder
from mlp_transformer_decoder import MLPTransformerDecoder
from train_eval_mlp_decoder import train_model, eval_model

# External
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import TransformerDecoderLayer, TransformerDecoder
import torch.nn as nn
import torch


def create_train_val_test_dataloaders(root_dir, batch_size):
    parser = ParseEquationsWeights(root_dir)
    embedded_equations, weights = parser.equations_weights_numpy_array()
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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader


def create_model():
    # Encoder parameters
    encoder_input = 128
    encoder_hidden = 256
    encoder_output = 512
    # Decoder parameters
    n_layers_decoder = 20
    # Decoder layer parameters
    n_heads_decoder = 8
    decoder_layer_activation_function = "relu"
    decoder_layer_feed_forward_dimension = 1024
    decoder_layer_dropout = 0.3
    # Building the model
    encoder = MLPEncoder(
        input_size=encoder_input, hidden_size=encoder_hidden, output_size=encoder_output
    )
    decoder_layer = TransformerDecoderLayer(
        d_model=encoder_output,
        nhead=n_heads_decoder,
        activation=decoder_layer_activation_function,
        dim_feedforward=decoder_layer_feed_forward_dimension,
        dropout=decoder_layer_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer=decoder_layer, num_layers=n_layers_decoder
    )

    mlp_transformer_decoder = MLPTransformerDecoder(encoder=encoder, decoder=decoder)
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
        optimizer, T_0=20, eta_min=1e-05
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
