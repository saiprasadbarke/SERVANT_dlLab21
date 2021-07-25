from settings import DEVICE, EOS_IDX, BOS_IDX
import torch.nn as nn
from typing import List
from vocab_transform import VOCAB_TRANSFORM
from beam_search import Translator
from torch import transpose, device, tensor
from symbolic_regression_transformer import SymbolicRegressionTransformer
from torch.utils.data import DataLoader
from greedy_search import greedy_decode

# actual function to translate input sentence into target language
def generate_equation(model: nn.Module, src_list: List, search_type: str):
    model.eval()
    src = tensor(src_list).view(1, -1)
    if search_type == "greedy":
        tgt_tokens = greedy_decode(model, src, start_symbol=BOS_IDX).flatten()
    elif search_type == "beam":
        model.to(device("cpu"))
        translator = Translator(model, beam_size=5, max_seq_len=20)
        tgt_tokens = translator.translate_sentence(src)
    return (
        " ".join(VOCAB_TRANSFORM.lookup_tokens(list(tgt_tokens.cpu().numpy())))
        .replace("<bos>", "")
        .replace("<eos>", "")
    )


def test_model(
    test_dataloader: DataLoader, model: SymbolicRegressionTransformer, search_type: str
):
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
            print(
                f"Predicted equation : {generate_equation(model, weights, search_type)}"
            )
