import torch
import torch.nn as nn
import numpy as np
from Layers import EncoderLayer, DecoderLayer

def get_pad_mask(seq, pad_idx):
    # seq shape: (batch_size, num_steps)
    batch_size, num_steps = seq.shape
    
    return (seq != pad_idx).view(batch_size, 1, 1, num_steps)

def get_subsequent_mask(seq):
    # seq shape: (batch_size, tgt_len)
    batch_size, tgt_len = seq.shape
    mask = torch.ones(tgt_len, tgt_len, dtype=torch.int)
    mask = (1 - torch.triu(mask, 1)).repeat(batch_size, 1, 1)
    # mask shape: (batch_size, tgt_len, tgt_len)
    return mask.unsqueeze(1) # mask shape: (batch_size, 1, tgt_len, tgt_len)


class PositionalEncoding(nn.Module):
    pos_table: torch.Tensor
    
    def __init__(self, d_model, n_posiion=200) -> None:
        super().__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(d_model, n_posiion))
    
    def _get_sinusoid_encoding_table(self, d_model, n_position):
        def get_position_angle_vec(position):
            # i = 0, d = 0/1, sin/cos
            # i = 1, d = 2/3, sin/cos
            return [position / np.power(10000, 2 * (d // 2) / d_model) for d in range(d_model)]
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        # 每个pos中，奇数维度使用sin，偶数维度使用cos
        # dim=0 代表pos_i; dim=1 代表d;
        sinusoid_table[:, ::2] = np.cos(sinusoid_table[:, ::2])
        sinusoid_table[:, 1::2] = np.sin(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0) # sinusoid_table shape: (1, n_position, d_model)
    
    def forward(self, x):
        # x shape: (batch_size, num_steps, d_model)
        return x + self.pos_table[:, :x.shape[1]].clone().detach()


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_k, d_v, d_inner, n_layers, dropout=0.1) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.position_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, n_head, d_k, d_v, d_inner) for _ in range(n_layers)])
    
    def forward(self, src_seq, mask=None, return_attn_weights=False):
        attn_weights_list = []
        # src_seq shape: (batch_size, src_len)
        enc_output = self.emb(src_seq)
        # enc_input shape: (batch_size, src_len, d_model)
        enc_output = self.dropout(self.position_enc(enc_output))

        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, attn_weights = enc_layer(enc_output, mask)
            if return_attn_weights:
                attn_weights_list.append(attn_weights)
        
        if return_attn_weights:
            return enc_output, attn_weights_list
        return enc_output,


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_k, d_v, d_inner, n_layers, dropout=0.1) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.position_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, n_head, d_k, d_v, d_inner) for _ in range(n_layers)])
    
    def forward(self, tgt_seq, enc_output, dec_mask=None, enc_mask=None, return_attn_weights=False):
        dec_attn_weights_list, dec_enc_attn_weights_list = [], []
        # tgt_seq shape: (batch_size, tgt_len)
        dec_output = self.emb(tgt_seq)
        # dec_output shape: (batch_size, tgt_len, d_model)
        dec_output = self.dropout(self.position_enc(dec_output))

        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_attn_weights, dec_enc_attn_weights= dec_layer(dec_output, enc_output, dec_mask=dec_mask, enc_mask=enc_mask)
            if return_attn_weights:
                dec_attn_weights_list.append(dec_attn_weights)
                dec_enc_attn_weights_list.append(dec_enc_attn_weights)
        
        if return_attn_weights:
            return dec_output, dec_attn_weights, dec_enc_attn_weights
        return dec_output,


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_head, d_k, d_v, d_inner, n_layers, pad_idx, dropout=0.1) -> None:
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_head, d_k, d_v, d_inner, n_layers, dropout=dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_head, d_k, d_v, d_inner, n_layers, dropout=dropout)
        self.dense = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx
    
    def forward(self, src_seq, tgt_seq):
        enc_mask = get_pad_mask(src_seq, pad_idx=self.pad_idx)
        dec_mask = get_pad_mask(tgt_seq, pad_idx=self.pad_idx) & get_subsequent_mask(tgt_seq)

        enc_output, *_ = self.encoder(src_seq, mask=enc_mask)
        dec_output, *_ = self.decoder(tgt_seq, enc_output, dec_mask=dec_mask, enc_mask=enc_mask)

        seq_logit = self.dense(dec_output)

        return seq_logit.view(-1, seq_logit.shape[2])


if __name__ == '__main__':
    src_vocab_size, tgt_vocab_size, batch_size, num_steps = 184, 201, 5, 15
    d_model, n_head, d_k, d_v, d_inner, n_layers = 512, 8, 64, 64, 2048, 6

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_head, d_k, d_v, d_inner, n_layers, 1)
    
    src_seq = torch.tensor([[ 47,  48,   5,   3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1],
        [ 13,  25,   4,   3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1],
        [ 86,   8,   4,   3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1],
        [111,  12,   4,   3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1],
        [105,   6,   9,  11,   3,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1]])
    tgt_seq = torch.tensor([[17,  0,  5,  3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 0,  5,  3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 0,  4,  3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [ 0,  4,  3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [87,  0,  0,  9,  3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
    
    seq_logit = model(src_seq, tgt_seq)
    print(seq_logit.shape)