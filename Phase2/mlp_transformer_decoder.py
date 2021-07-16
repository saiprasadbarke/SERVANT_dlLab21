import torch.nn as nn
from mlp_encoder import MLPEncoder
from transformer_decoder import TransformerDecoderLol


class MLPTransformerDecoder(nn.Module):
    def __init__(self, encoder: MLPEncoder, decoder: TransformerDecoderLol):
        super(MLPTransformerDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.log_softmax = nn.LogSoftmax()

    def forward(self, src, tgt):
        return self.log_softmax(self.decode(self.encode(src), tgt))

    def encode(self, src):
        return self.encoder(src)

    def decode(self, target, memory):
        return self.decoder(target, memory)
