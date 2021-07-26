from torch import ones, triu
from settings import PAD_IDX, DEVICE


def generate_square_subsequent_mask(sz):
    mask = (triu(ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(tgt):
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return tgt_mask, tgt_padding_mask
