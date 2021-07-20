import torch
import torch.nn as nn


if __name__ == "__main__":
    decoder_layer = nn.TransformerDecoderLayer(d_model=17, nhead=17)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
    memory = torch.rand(30, 10, 17)  #  10, 8, 16
    tgt = torch.rand(20, 10, 17)
    out = transformer_decoder(tgt, memory)
    print(nn.out[:, 0, :])
    print("lol")
