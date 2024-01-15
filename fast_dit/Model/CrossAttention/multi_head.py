import torch.nn as nn
from math import sqrt

from fast_dit.Model.CrossAttention.attention import Attention


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size  # 输入维度
        self.all_head_size = all_head_size  # 输出维度
        self.num_heads = head_num  # 注意头的数量
        self.h_size = all_head_size // head_num

        assert all_head_size % head_num == 0

        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size, self.all_head_size)
        print(self.linear_k, self.linear_q, self.linear_v)

    def forward(self, x, y):
        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = (
            self.linear_q(x)
            .view(batch_size, -1, self.num_heads, self.h_size)
            .transpose(1, 2)
        )

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = (
            self.linear_k(y)
            .view(batch_size, -1, self.num_heads, self.h_size)
            .transpose(1, 2)
        )

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = (
            self.linear_v(y)
            .view(batch_size, -1, self.num_heads, self.h_size)
            .transpose(1, 2)
        )

        attention = Attention()(q_s, k_s, v_s)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.h_size)
        )

        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output
