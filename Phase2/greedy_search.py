from settings import DEVICE, EOS_IDX
from torch import long, bool, ones, max, cat
from masking import generate_square_subsequent_mask


def greedy_decode(model, src, start_symbol, max_len=20):
    src = src.to(DEVICE)
    memory = model.encode(src)
    ys = ones(1, 1).fill_(start_symbol).type(long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(bool)).to(DEVICE)
        out = model.decode(memory, ys, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = max(prob, dim=1)
        next_word = next_word.item()

        ys = cat([ys, ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys
