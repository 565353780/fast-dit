import torch

from fast_dit.Model.diffusion import create_diffusion
from fast_dit.Model.asdf_dit import ASDFDiT

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def test():
    # Setup PyTorch:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = "cpu"

    asdf_channel = 100
    asdf_dim = 40
    context_dim = 40
    num_heads = 6
    head_dim = 64
    depth = 12
    num_sampling_steps = 2

    model = ASDFDiT(asdf_channel, asdf_dim, context_dim, num_heads, head_dim, depth).to(
        device
    )

    # state_dict = find_model(ckpt_path)
    # model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(num_sampling_steps))

    # Labels to condition the model with (feel free to change):

    batch_size = 1
    z = torch.randn(batch_size, 1, asdf_channel, asdf_dim, device=device)
    y = torch.randn(batch_size, 1, asdf_channel, context_dim, device=device)

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
        device=device,
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    print(samples.shape)
    return True
