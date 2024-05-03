import os
import torch
from time import time
from copy import deepcopy
from accelerate import Accelerator
from torch.utils.data import DataLoader

from fast_dit.Dataset.mash import MashDataset
from fast_dit.Model.mash_dit import MashDiT
from fast_dit.Model.diffusion import create_diffusion
from fast_dit.Method.train import update_ema, requires_grad, create_logger
from fast_dit.Method.time import getCurrentTime
from fast_dit.Module.logger import Logger

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Trainer(object):
    def __init__(self) -> None:
        self.dataset_folder_path = '/home/chli/Dataset/'
        self.epochs = 100000
        self.global_batch_size = 1000
        self.num_workers = 4
        self.log_every = 1
        self.ckpt_every = 300000
        self.lr = 1e-4

        self.mash_channel = 400
        self.mash_dim = 22
        self.context_dim = 768
        self.num_heads = 6
        self.head_dim = 64
        self.depth = 12

        self.diffusion_steps=36

        # assert torch.cuda.is_available(), "Training currently requires at least one GPU."

        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        if self.accelerator.is_main_process:
            current_time = getCurrentTime()
            self.output_folder_path = './output/' + current_time + '/'
            log_folder_path = './logs/' + current_time + '/'
            os.makedirs(self.output_folder_path, exist_ok=True)
            os.makedirs(log_folder_path, exist_ok=True)
            self.logger = Logger(log_folder_path)
        return

    def trainStep(self) -> bool:
        return True

    def train(self) -> bool:
        # Setup an experiment folder:
        if self.accelerator.is_main_process:
            logger = create_logger(self.output_folder_path)
            logger.info(f"Experiment directory created at {self.output_folder_path}")

        # Create model:
        model = MashDiT(self.mash_channel, self.mash_dim, self.context_dim, self.num_heads, self.head_dim, self.depth).to(
            self.device
        )
        # Note that parameter initialization is done within the DiT constructor
        model = model.to(self.device)
        # Create an EMA of the model for use after training
        ema = deepcopy(model).to(self.device)
        requires_grad(ema, False)
        # default: 1000 steps, linear noise schedule
        diffusion = create_diffusion(timestep_respacing="", diffusion_steps=self.diffusion_steps)
        if self.accelerator.is_main_process:
            logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0)

        # Setup data:
        dataset = MashDataset(self.dataset_folder_path)
        loader = DataLoader(
            dataset,
            batch_size=int(self.global_batch_size // self.accelerator.num_processes),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        if self.accelerator.is_main_process:
            logger.info(f"Dataset contains {len(dataset):,} Mashes")

        # Prepare models for training:
        # Ensure EMA is initialized with synced weights
        update_ema(ema, model, decay=0)
        model.train()  # important! This enables embedding dropout for classifier-free guidance
        ema.eval()  # EMA model should always be in eval mode
        model, opt, loader = self.accelerator.prepare(model, opt, loader)

        # Variables for monitoring/logging purposes:
        train_steps = 0
        log_steps = 0
        running_loss = 0
        start_time = time()

        if self.accelerator.is_main_process:
            logger.info(f"Training for {self.epochs} self.epochs...")
        for epoch in range(self.epochs):
            if self.accelerator.is_main_process:
                logger.info(f"Beginning epoch {epoch}...")
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                # x = x.squeeze(dim=1)
                # y = y.squeeze(dim=1)
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=self.device)
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                opt.zero_grad()
                self.accelerator.backward(loss)
                opt.step()
                update_ema(ema, model)

                # Log loss values:
                running_loss += loss.item()
                log_steps += 1
                train_steps += 1
                if train_steps % self.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=self.device)
                    avg_loss = avg_loss.item() / self.accelerator.num_processes
                    if self.accelerator.is_main_process:
                        logger.info(
                            f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                        )

                        self.logger.addScalar('Train/loss', avg_loss)
                        self.logger.addScalar('Train/StepSec', steps_per_sec)
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save DiT checkpoint:
                if train_steps % self.ckpt_every == 0 and train_steps > 0:
                    if self.accelerator.is_main_process:
                        checkpoint = {
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                        }
                        checkpoint_path = f"{self.output_folder_path}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

        model.eval()  # important! This disables randomized embedding dropout
        # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

        if self.accelerator.is_main_process:
            logger.info("Done!")

        return True
