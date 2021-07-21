import torch.nn as nn
from torch.nn.modules.transformer import TransformerDecoder
from mlp_encoder import MLPEncoder


class SymbolicRegressionTransformer(nn.Module):
    def __init__(
        self,
        encoder: MLPEncoder,
        decoder: TransformerDecoder,
        tgt_embed,
        generator,
    ):
        super(SymbolicRegressionTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoded_mem = self.encode(src, src_mask)
        decoded_seq = self.decode(encoded_mem, src_mask, tgt, tgt_mask)
        return decoded_seq

    def encode(self, src):
        encoded_mem = self.encoder(src)
        return encoded_mem

    def decode(self, memory, tgt, tgt_mask):
        target_embedded = self.tgt_embed(tgt)
        decoded_seq = self.decoder(
            target_embedded,
            memory,
            tgt_mask,
        )
        return decoded_seq
