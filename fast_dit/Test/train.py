import os
import torch
import argparse
from time import time
from glob import glob
from copy import deepcopy
from accelerate import Accelerator
from torch.utils.data import DataLoader

from fast_dit.Dataset.asdf import ASDFDataset
from fast_dit.Model.asdf_dit import ASDFDiT
from fast_dit.Model.diffusion import create_diffusion
from fast_dit.Method.train import update_ema, requires_grad, create_logger

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="output")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=0)
    # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()

    # assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    accelerator.device = "cpu"
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        # Make results folder (holds all experiment subfolders)
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        model_string_name = "ASDF"
        # Create an experiment folder
        experiment_dir = (
            f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        )
        # Stores saved model checkpoints
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    asdf_channel = 100
    asdf_dim = 40
    context_dim = 40
    num_heads = 6
    head_dim = 64
    depth = 12

    model = ASDFDiT(asdf_channel, asdf_dim, context_dim, num_heads, head_dim, depth).to(
        device
    )
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    # Create an EMA of the model for use after training
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(timestep_respacing="")
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    dataset = ASDFDataset("1")
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    # Ensure EMA is initialized with synced weights
    update_ema(ema, model, decay=0)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if accelerator.is_main_process:
        logger.info("Done!")

    return True
