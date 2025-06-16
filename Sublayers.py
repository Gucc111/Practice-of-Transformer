import torch
from torch import nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn_weights = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # attn_weights shape: (batch_size, n_head, num_steps, num_steps)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask==0, -1e9)
        
        attn_weights = self.dropout(F.softmax(attn_weights, dim=-1))
        output = torch.matmul(attn_weights, v)
        # output shape: (batch_size, n_head, num_steps, d_v)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1) -> None:
        super().__init__()
        self.w_q = nn.Linear(d_model, n_head * d_k)
        self.w_k = nn.Linear(d_model, n_head * d_k)
        self.w_v = nn.Linear(d_model, n_head * d_v)
        self.attn = ScaledDotProductAttention(temperature=d_k**0.5)
        self.dense = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.n_head = n_head
    
    def forward(self, q, k, v, mask=None):
        # q/k/v shape: (batch_size, num_steps, d_model)
        batch_size, len_q = q.shape[:2]
        len_k, len_v = k.shape[1], v.shape[1]
        residual = q

        q = self.w_q(q).view(batch_size, len_q, self.n_head, -1)
        k = self.w_k(k).view(batch_size, len_k, self.n_head, -1)
        v = self.w_v(v).view(batch_size, len_v, self.n_head, -1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q/k/v shape: (batch_size, n_head, num_steps, d_k/d_v)
        q, attn_weights = self.attn(q, k, v, mask=mask)
        # q shape: (batch_size, n_head, num_steps, d_v)
        # attn_weights: (batch_size, n_head, num_steps, num_steps)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        # q shape: (batch_size, num_steps, n_head * d_v)
        q = self.dropout(self.dense(q))
        # q shape: (batch_size, num_steps, d_model)
        q += residual
        
        q = self.layer_norm(q)

        return q, attn_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout = 0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, x):
        # x shape: (batch_size, num_steps, d_model)
        residual = x
        x = self.w_2(F.relu((self.w_1(x))))

        x = self.dropout(x)
        
        x += residual

        x = self.layer_norm(x)

        return x


if __name__ == '__main__':
    batch_size, num_steps = 64, 5
    d_model, n_head, d_k, d_v, d_inner = 512, 8, 64, 64, 2048
    attn = MultiHeadAttention(d_model, n_head, d_k, d_v)
    q = torch.randn((batch_size, num_steps, d_model))
    output, attn_weights = attn(q, q, q)
    fnn = PositionwiseFeedForward(d_model, d_inner)
    output = fnn(output)
    print(output.shape)
    print(attn_weights.shape)