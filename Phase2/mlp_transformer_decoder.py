import torch.nn as nn
from mlp_encoder import MLPEncoder
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class MLPTransformerDecoder(nn.Module):
    def __init__(self, encoder: MLPEncoder, decoder: TransformerDecoder):
        super(MLPTransformerDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.log_softmax = nn.LogSoftmax()

    def forward(self, src, tgt):
        encoded_mem = self.encode(src)
        decoded_seq = self.decode(tgt, encoded_mem)
        softmax_output = self.log_softmax(decoded_seq)
        return softmax_output

    def encode(self, src):
        encoded_mem = self.encoder(src)
        return encoded_mem

    def decode(self, target, memory):
        decoded_seq = self.decoder(target, memory)
        return decoded_seq
