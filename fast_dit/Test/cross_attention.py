import torch
from fast_dit.Model.CrossAttention.multi_head import MultiHeadCrossAttention


def test():
    feature_length = 40
    hidden_feature_length = 256
    sub_space_num = 8

    multi_head_cross_attention = MultiHeadCrossAttention(
        feature_length, hidden_feature_length, sub_space_num
    )
    multi_head_cross_attention.print()

    features = torch.rand(10, 3, feature_length)
    conditions = torch.rand(20, 2, feature_length)

    cross_output = multi_head_cross_attention(features, conditions)
    print(cross_output.shape)
    return True
