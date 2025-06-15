import torch
from torch import nn
import torch.nn.functional as F
from Sublayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, dropout=0.1) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout=dropout)
        self.pos_fnn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    
    def forward(self, enc_input, mask=None):
        # enc_input shape: (batch_size, src_len, d_model)
        enc_output, attn_weights = self.attn(enc_input, enc_input, enc_input, mask)
        # enc_output shape: (batch_size, src_len, d_model)
        # attn_weights: (batch_size, n_head, src_len, src_len)
        enc_output = self.pos_fnn(enc_output)
        # enc_output shape: (batch_size, src_len, d_model)
        return enc_output, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, dropout=0.1) -> None:
        super().__init__()
        self.dec_attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout=dropout)
        self.pos_fnn = PositionwiseFeedForward(d_model, d_inner)
    
    def forward(self, dec_input, enc_output, dec_mask=None, enc_mask=None):
        dec_output, dec_attn_weights = self.dec_attn(dec_input, dec_input, dec_input, mask=dec_mask)
        # dec_output shape: (batch_size, tgt_len, d_model)
        # dec_attn_weights: (batch_size, n_head, tgt_len, tgt_len)
        # enc_output shape: (batch_size, src_len, d_model)
        dec_output, dec_enc_attn_weights = self.enc_attn(dec_output, enc_output, enc_output, mask=enc_mask)
        # dec_output shape: (batch_size, tgt_len, d_model)
        # dec_enc_attn_weights: (batch_size, n_head, tgt_len, src_len)
        return dec_output, dec_attn_weights, dec_enc_attn_weights


if __name__ == '__main__':
    batch_size, num_steps = 64, 5
    d_model, n_head, d_k, d_v, d_inner = 512, 8, 64, 64, 2048
    
    enclayer = EncoderLayer(d_model, n_head, d_k, d_v, d_inner)
    enc_input = torch.randn(batch_size, num_steps, d_model)
    enc_output, _ = enclayer(enc_input)
    
    declayer = DecoderLayer(d_model, n_head, d_k, d_v, d_inner)
    dec_input = torch.randn(batch_size, num_steps, d_model)
    dec_output, *_ = declayer(dec_input, enc_output)
    print(enc_output.shape)
    print(dec_output.shape)