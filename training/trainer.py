"""
Omni-ST: Training Pipeline — Main Trainer
==========================================
Configurable trainer for all 3 training stages:
  - Stage 1: Modality-specific pretraining
  - Stage 2: Cross-modal contrastive alignment (CLIP-style)
  - Stage 3: Instruction-conditioned task fine-tuning

Features:
  - Mixed precision (torch.cuda.amp)
  - Gradient clipping and accumulation
  - Weights & Biases experiment tracking
  - Checkpoint save/resume
  - Multi-GPU via HuggingFace Accelerate
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class OmniSTTrainer:
    """
    General-purpose trainer for Omni-ST models.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer | None
        If None, a default AdamW is created.
    scheduler : torch.optim.lr_scheduler._LRScheduler | None
    criterion : nn.Module | callable | None
    device : str
    use_amp : bool  mixed precision
    grad_clip : float  gradient norm clipping value
    grad_accum_steps : int  accumulate gradients over N micro-batches
    log_interval : int  log every N batches
    checkpoint_dir : str
    wandb_config : dict | None  Weights & Biases config (project, name, tags)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        criterion: Optional[Any] = None,
        device: str = "cuda",
        use_amp: bool = True,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        log_interval: int = 50,
        checkpoint_dir: str = "checkpoints",
        wandb_config: Optional[Dict] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optimizer or AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = scheduler
        self.criterion = criterion or nn.MSELoss()
        self.scaler = GradScaler(enabled=use_amp and torch.cuda.is_available())
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_val_loss = float("inf")

        # Weights & Biases
        self.use_wandb = wandb_config is not None
        if self.use_wandb:
            try:
                import wandb
                wandb.init(**wandb_config)
                wandb.watch(model, log="gradients", log_freq=100)
            except ImportError:
                print("wandb not installed. Skipping W&B logging.")
                self.use_wandb = False

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            batch = self._to_device(batch)

            with autocast(enabled=self.use_amp):
                loss = self._forward_pass(batch)
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.grad_accum_steps

            if step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
                if self.use_wandb:
                    import wandb
                    wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": self.global_step})

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in dataloader:
            batch = self._to_device(batch)
            with autocast(enabled=self.use_amp):
                loss = self._forward_pass(batch)
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        save_best: bool = True,
        early_stop_patience: int = 20,
    ) -> Dict[str, list]:
        history = {"train_loss": [], "val_loss": []}
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.evaluate(val_loader)
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if self.use_wandb:
                import wandb
                wandb.log({"epoch/train_loss": train_loss, "epoch/val_loss": val_loss, "epoch": epoch})

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    self.save_checkpoint(epoch, val_loss, tag="best")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch} (patience={early_stop_patience})")
                    break

        return history

    # ------------------------------------------------------------------
    # Abstract forward pass (override in subclasses)
    # ------------------------------------------------------------------

    def _forward_pass(self, batch: Dict) -> torch.Tensor:
        """
        Default forward pass. Subclasses must override for task-specific logic.
        """
        raise NotImplementedError("Override _forward_pass in a task-specific trainer subclass.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _to_device(self, batch: Dict) -> Dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "latest") -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "global_step": self.global_step,
        }
        path = self.checkpoint_dir / f"omni_st_{tag}.pt"
        torch.save(ckpt, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
        return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Default Scheduler Factory
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 500,
    total_steps: int = 10000,
) -> SequentialLR:
    """Warmup then cosine annealing scheduler."""
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
