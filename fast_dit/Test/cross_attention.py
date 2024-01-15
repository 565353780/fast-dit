import torch
from fast_dit.Model.CrossAttention.cross import CrossAttention
from fast_dit.Model.CrossAttention.multi_head import MultiHeadCrossAttention


def test_cross_attention():
    query_dim = 40
    context_dim = 30
    heads = 17
    dim_head = 3
    dropout = 0.0

    cross_attention = CrossAttention(query_dim, context_dim, heads, dim_head, dropout)

    features = torch.rand(10, 3, query_dim)
    conditions = torch.rand(10, 2, context_dim)

    cross_output = cross_attention(features, conditions)
    print(cross_output.shape)
    return True


def test_multi_head_cross_attention():
    query_dim = 40
    context_dim = 30
    heads = 17
    dim_head = 3

    multi_head_cross_attention = MultiHeadCrossAttention(
        query_dim, context_dim, heads, dim_head
    )

    features = torch.rand(10, 3, query_dim)
    conditions = torch.rand(20, 2, context_dim)

    cross_output = multi_head_cross_attention(features, conditions)
    print(cross_output.shape)
    return True


def test():
    test_cross_attention()
    test_multi_head_cross_attention()
    return True
