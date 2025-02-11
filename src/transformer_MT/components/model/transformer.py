import torch
from torch import nn
from src.transformer_MT.components.model.positional_encoding import PositionalEncoding
from src.transformer_MT.components.model.encoder import Encoder
from src.transformer_MT.components.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model).to(device)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model).to(device)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length).to(device)

        self.encoder_layers = nn.ModuleList([Encoder(d_model, num_heads, d_ff, dropout).to(device) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([Decoder(d_model, num_heads, d_ff, dropout).to(device) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size).to(device)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def generate_mask(self, src, tgt):
        device = src.device
        src_mask = (src != 1).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 1).unsqueeze(1).unsqueeze(3)
        seq_length_src = src.size(1) 
        seq_length_tgt = tgt.size(1) 
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length_tgt, seq_length_tgt), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask  
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src.long()).to(self.device)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt.long()).to(self.device)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output