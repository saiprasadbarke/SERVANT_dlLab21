import torch.nn as nn
from torch.nn.modules.transformer import TransformerDecoder
from mlp_encoder import MLPEncoder
from token_embedding import TokenEmbedding
from positional_encoding import PositionalEncoding
from generator import Generator


class SymbolicRegressionTransformer(nn.Module):
    def __init__(
        self,
        encoder: MLPEncoder,
        decoder: TransformerDecoder,
        target_token_embedding: TokenEmbedding,
        target_positional_embedding: PositionalEncoding,
        generator: Generator,
    ):
        super(SymbolicRegressionTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_token_embedding = target_token_embedding
        self.target_positional_embedding = target_positional_embedding
        self.generator = generator

    def forward(self, src, tgt, tgt_mask):
        encoded_mem = self.encode(src)
        decoded_seq = self.decode(encoded_mem, tgt, tgt_mask)
        return decoded_seq

    def encode(self, src):
        encoded_mem = self.encoder(src)
        return encoded_mem

    def decode(self, memory, tgt, tgt_mask):
        target_embedded = self.tgt_embed(tgt)
        decoded_seq = self.decoder(
            tgt=target_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
        )
        return decoded_seq
