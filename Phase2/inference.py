import torch
from settings import DEVICE, EOS_IDX, BOS_IDX
from masking import generate_square_subsequent_mask
from typing import List
from vocab_transform import VOCAB_TRANSFORM


def greedy_decode(model, src, start_symbol, max_len=20):
    src = src.to(DEVICE)

    memory = model.encode(src)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(memory, ys, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_list: List):
    model.eval()
    src = torch.tensor(src_list).view(1, -1)
    tgt_tokens = greedy_decode(model, src, start_symbol=BOS_IDX).flatten()
    return (
        " ".join(VOCAB_TRANSFORM.lookup_tokens(list(tgt_tokens.cpu().numpy())))
        .replace("<bos>", "")
        .replace("<eos>", "")
    )
