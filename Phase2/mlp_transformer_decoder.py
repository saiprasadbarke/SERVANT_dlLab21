import torch.nn as nn
from mlp_encoder import MLPEncoder
from transformer_decoder_teacher_forcing import TransformerDecoderTeacherForcing


class MLPTransformerDecoder(nn.Module):
    def __init__(self, encoder: MLPEncoder, decoder: TransformerDecoderTeacherForcing):
        super(MLPTransformerDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        encoded_mem = self.encode(src)
        decoded_seq = self.decode(tgt, encoded_mem)
        return decoded_seq

    def encode(self, src):
        encoded_mem = self.encoder(src)
        return encoded_mem

    def decode(self, target, memory):
        decoded_seq = self.decoder(target, memory)
        return decoded_seq
