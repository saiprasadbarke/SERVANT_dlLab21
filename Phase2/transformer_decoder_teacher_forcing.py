import torch.nn as nn
import copy
from torch import tensor, unsqueeze, Tensor, cat
import random


class TransformerDecoderTeacherForcing(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoderTeacherForcing, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        r"""Pass the inputs through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).

        Shape:
            see the docs in Transformer class.
        """
        outputs = []
        teacher_forcing_ratio = 0.5
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for idx, mod in enumerate(self.layers):
                print(tgt[idx].unsqueeze(0).shape)
                print(tgt[idx].unsqueeze(0))
                output = mod(tgt[idx].unsqueeze(0), memory)
                outputs.append(output)
            return cat(outputs)
        else:
            output = tgt[0].unsqueeze(0)
            for mod in self.layers:
                print(output.shape)
                print(output)
                output = mod(output, memory)
                outputs.append(output)
            return cat(outputs)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
