import torch
import torch.nn as nn

from fast_dit.Model.timestep_embedder import TimestepEmbedder
from fast_dit.Model.label_embedder import LabelEmbedder
from fast_dit.Model.DiT.block import DiTBlock
from fast_dit.Model.DiT.final_layer import FinalLayer


class ASDFDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        asdf_dims=[100, 40],
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        assert (asdf_dims[1] * hidden_size) % (patch_size**2) == 0

        self.x_embedder = nn.Linear(
            asdf_dims[1],
            int(asdf_dims[1] * hidden_size / patch_size / patch_size),
            bias=False,
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

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
        y: (N,) tensor of class labels
        """
        print("DiT forward:")
        print("x:", x.shape)
        print("t:", t.shape)
        print("y:", y.shape)
        N, _, H, W = x.shape
        # (N, T, D), where T = H * W / patch_size ** 2
        x = self.x_embedder(x)
        print("after x_embedder, x:", x.shape)
        assert (H * W) % (self.patch_size**2) == 0
        T = int(H * W / self.patch_size / self.patch_size)
        print("start reshape:", x.shape, "-->", N, T, self.hidden_size)
        x = x.reshape(N, T, self.hidden_size)
        print("after reshape, x:", x.shape)
        t = self.t_embedder(t)  # (N, D)
        print("after t_embedder, t:", t.shape)
        y = self.y_embedder(y, self.training)  # (N, D)
        print("after y_embedder, y:", y.shape)
        c = t + y  # (N, D)
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(block), x, c
            )  # (N, T, D)
        print("after blocks, x:", x.shape)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        print("after final_layer, x:", x.shape)
        x = x.reshape(N, -1, H, W)  # (N, out_channels, H, W)
        print("after reshape, x:", x.shape)
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
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
