# Local
from equations_weights_dataset import EquationsWeightsDataset
from parse_equations_weights import ParseEquationsWeights
from mlp_encoder import MLPEncoder
from transformer_decoder import TransformerDecoderImpl
from mlp_transformer_decoder import MLPTransformerDecoder

# External
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def create_train_val_test_dataloaders(root_dir):
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

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":

    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = create_train_val_test_dataloaders(
        "./network_wts_eqs_dataset/ntwrk_wts_eqs_1000.json"
    )
    print()
