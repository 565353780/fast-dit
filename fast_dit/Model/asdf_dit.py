import torch
import torch.nn as nn

from fast_dit.Model.DiT.final_layer import FinalLayer
from fast_dit.Model.DiT.asdf_block import ASDFDiTBlock
from fast_dit.Model.CrossAttention.cross import CrossAttention
from fast_dit.Model.timestep_embedder import TimestepEmbedder


class ASDFDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        asdf_channel=100,
        asdf_dim=40,
        context_dim=30,
        num_heads=6,
        head_dim=64,
        depth=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ):
        super().__init__()
        self.asdf_channel = asdf_channel
        self.asdf_dim = asdf_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.learn_sigma = learn_sigma
        self.out_channels = 2 if learn_sigma else 1
        self.hidden_dim = num_heads * head_dim

        self.xy_cross_attention = CrossAttention(
            self.hidden_dim, self.hidden_dim, self.num_heads, self.head_dim, 0.0
        )
        self.ty_cross_attention = CrossAttention(
            self.hidden_dim, self.hidden_dim, self.num_heads, self.head_dim, 0.0
        )
        self.x_embedder = nn.Linear(self.asdf_dim, self.hidden_dim, bias=False)
        self.y_embedder = nn.Linear(self.context_dim, self.hidden_dim, bias=False)
        self.t_embedder = TimestepEmbedder(self.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                ASDFDiTBlock(self.hidden_dim, self.num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            self.hidden_dim, 1, self.out_channels * self.asdf_dim
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of contexts
        """
        B = x.shape[0]

        x = x.reshape(B, self.asdf_channel, self.asdf_dim)
        y = y.reshape(B, self.asdf_channel, self.context_dim)

        x = self.x_embedder(x)
        y = self.y_embedder(y)
        t = self.t_embedder(t)

        t = t.reshape(B, 1, self.hidden_dim)

        xy = self.xy_cross_attention(x, y)
        ty = self.ty_cross_attention(t, y)
        x = x + xy
        c = t + ty

        t = t.reshape(B, self.hidden_dim)
        c = c.reshape(B, self.hidden_dim)

        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(block), x, c
            )  # (N, T, D)
        x = self.final_layer(x, t)  # (N, T, out_channels)
        x = x.reshape(B, self.out_channels, self.asdf_channel, self.asdf_dim)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps = model_out[:, 0]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
