import torch
from math import sqrt

from ma_sh.Model.mash import Mash

from fast_dit.Model.diffusion import create_diffusion
from fast_dit.Model.dit import DiT
from fast_dit.Method.io import find_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Sampler(object):
    def __init__(self, model_file_path: str) -> None:
        self.model_file_path = model_file_path

        self.mash_channel = 22
        self.mash_dim = 400
        self.context_dim = 768
        self.patch_size = 2
        self.num_heads = 6
        self.depth = 12

        self.image_dim = int(sqrt(self.mash_dim))
        assert self.image_dim ** 2 == self.mash_dim

        self.device = "cuda"

        self.diffusion_steps=36

        self.sh_2d_degree = 3
        self.sh_3d_degree = 2

        torch.manual_seed(0)
        return

    def toInitialMashModel(self) -> Mash:
        mash_model = Mash(
            self.mash_dim,
            self.sh_2d_degree,
            self.sh_3d_degree,
            dtype=torch.float32,
            device=self.device,
        )
        return mash_model

    @torch.no_grad()
    def sample(self, sample_num: int, category_id: int) -> torch.Tensor:
        model = DiT(self.image_dim, self.patch_size, self.mash_channel, self.context_dim, self.depth, self.num_heads).to(self.device)

        state_dict = find_model(self.model_file_path)
        model.load_state_dict(state_dict)
        model.eval()  # important!
        diffusion = create_diffusion("", diffusion_steps=self.diffusion_steps)

        # Labels to condition the model with (feel free to change):

        z = torch.randn(sample_num, self.mash_channel, self.image_dim, self.image_dim, device=self.device)
        y = torch.ones([sample_num], dtype=torch.long, device=self.device) * category_id

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y = torch.cat([y, y], 0)
        model_kwargs = dict(y=y, cfg_scale=4.0)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device,
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = samples.reshape(*samples.shape[:2], -1).permute(0, 2, 1)
        return samples
