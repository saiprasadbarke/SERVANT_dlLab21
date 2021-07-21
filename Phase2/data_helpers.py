from torch.nn.utils.rnn import pad_sequence
from typing import List, Iterable
from torch import cat, tensor
from torchtext.vocab import Vocab, build_vocab_from_iterator
from settings import BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX, special_symbols

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
def get_equations_tensor_transform(token_ids: List[int]):
    return cat((tensor([BOS_IDX]), tensor(token_ids), tensor([EOS_IDX])))


def get_equations_vocab_transform(equations_list: Iterable) -> Vocab:
    vocab_transform = build_vocab_from_iterator(
        equations_list,
        min_freq=1,
        specials=special_symbols,
        special_first=True,
    )
    vocab_transform.set_default_index(UNK_IDX)
    return vocab_transform


text_transform = sequential_transforms(
    get_equations_vocab_transform,  # Numericalization
    get_equations_tensor_transform,  # Add BOS/EOS and create tensor
)


# function to collate data samples into batch tesors
def collate_fn(batch):
    # src and tgt language text transforms to convert raw strings into tensors indices
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(text_transform(tgt_sample))
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
