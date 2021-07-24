from torch import transpose
from vocab_transform import VOCAB_TRANSFORM
from inference import generate_equation
from symbolic_regression_transformer import SymbolicRegressionTransformer
from torch.utils.data import DataLoader


def test_model(test_dataloader: DataLoader, model: SymbolicRegressionTransformer):
    for weights_list, equation_tokens in test_dataloader:
        equation_tokens = transpose(equation_tokens, 0, 1)
        weights_list, equation_tokens = (
            weights_list.tolist(),
            equation_tokens.tolist(),
        )

        for weights, equation in zip(weights_list, equation_tokens):
            equation_string = (
                "".join(VOCAB_TRANSFORM.lookup_tokens(equation))
                .replace("<bos>", "")
                .replace("<eos>", "")
                .replace("<pad>", "")
            )
            print(f"Ground Truth: {equation_string}")
            print(f"Predicted equation : {generate_equation(model, weights)}")
