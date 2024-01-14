import torch
import sys

sys.path.append("../fdit/")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    from fast_dit.Model.diffusion import create_diffusion
    from fast_dit.Model.dit import DiT

    # Setup PyTorch:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = "cpu"

    latent_size = 32
    num_classes = 1000
    num_sampling_steps = 2

    model = DiT(
        input_size=latent_size,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
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
    class_labels = [0 for _ in range(8)]

    # Create sampling noise:
    n = len(class_labels)
    # 100x40
    z = torch.randn(n, 4, 32, 32, device=device)
    # 100
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    # 200x40
    z = torch.cat([z, z], 0)
    # 100
    y_null = torch.tensor([1000] * n, device=device)
    # 200
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=4.0)

    # Sample images:
    # 200x40
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    # 100x40
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    print(samples.shape)
    return True


if __name__ == "__main__":
    main()
