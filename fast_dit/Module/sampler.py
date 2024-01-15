import torch

from a_sdf.Model.asdf_model import ASDFModel

from fast_dit.Model.diffusion import create_diffusion
from fast_dit.Model.asdf_dit import ASDFDiT
from fast_dit.Method.io import find_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Sampler(object):
    def __init__(self) -> None:
        self.model_file_path = './output/1/0000010.pt'
        self.asdf_channel = 100
        self.asdf_dim = 40
        self.context_dim = 40
        self.num_heads = 6
        self.head_dim = 64
        self.depth = 12
        self.device = "cuda"

        self.batch_size = 10
        self.num_sampling_steps = 2

        torch.manual_seed(0)
        return

    @torch.no_grad()
    def sample(self) -> bool:
        model = ASDFDiT(self.asdf_channel, self.asdf_dim, self.context_dim, self.num_heads, self.head_dim, self.depth).to(
            self.device
        )

        state_dict = find_model(self.model_file_path)
        model.load_state_dict(state_dict)
        model.eval()  # important!
        diffusion = create_diffusion(str(self.num_sampling_steps))

        # Labels to condition the model with (feel free to change):

        z = torch.randn(self.batch_size, 1, self.asdf_channel, self.asdf_dim, device=self.device)
        y = torch.randn(self.batch_size, 1, self.asdf_channel, self.context_dim, device=self.device)

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
        print(samples.shape)
        return True
