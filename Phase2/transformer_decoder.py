import torch.nn as nn


class TransformerDecoderLol(nn.Module):
    def __init__(self, memory, target, d_model, n_layers, n_heads):
        self.memory = memory
        self.target = target

        self.decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, n_layers)

    def forward(self):
        return self.decoder.forward(tgt=self.target, memory=self.memory)
