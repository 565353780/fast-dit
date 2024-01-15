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

    asdf_dims = [100, 40]
    depth = 12
    num_classes = 1000
    num_sampling_steps = 2

    model = ASDFDiT(
        asdf_dims=asdf_dims,
        patch_size=2,
        in_channels=1,
        hidden_size=asdf_dims[1] * depth,
        depth=depth,
        num_heads=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=num_classes,
        learn_sigma=True,
    ).to(device)

    # state_dict = find_model(ckpt_path)
    # model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(num_sampling_steps))

    # Labels to condition the model with (feel free to change):

    batch_size = 1
    z = torch.randn(batch_size, 1, asdf_dims[0], asdf_dims[1], device=device)
    y = torch.tensor([1 for _ in range(batch_size)], device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * batch_size, device=device)
    y = torch.cat([y, y_null], 0)
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
