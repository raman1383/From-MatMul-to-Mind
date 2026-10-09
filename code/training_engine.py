# training_engine.py

import math
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange

from data_loader import BatchLoader
from model_config import ModelConfig


CHECKPOINT_VERSION = 1

# Config fields that may legitimately differ between the original run and a
# resumed run. Everything else (architecture, vocab, context window, batch
# size...) must match or the resume is refused.
_RESUME_IGNORED_CONFIG_KEYS = {
    "model_name",
    "learning_rate",
    "save_path",
    "val_loss_history",
    "checkpoint_dir",
    "checkpoint_every_steps",
    "keep_last_n_checkpoints",
}


def _cross_entropy(logits, targets, reduction="mean"):
    return nn.functional.cross_entropy(
        rearrange(logits,
                  "batch_size seq_len vocab_size"
                  "->"
                  "(batch_size seq_len) vocab_size"),
        rearrange(targets,
                  "batch_size seq_len "
                  "->"
                  "(batch_size seq_len)"),
        reduction=reduction,
    )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
):
    """
    Linear warmup -> cosine decay to (min_lr_ratio * peak LR).

    `total_steps` is the length of the WHOLE planned run, not of one stage.
    The factor is a pure function of the step index, so resuming is exact:
    the scheduler's saved step counter puts us back on the same curve.
    Past `total_steps` the LR stays at the floor.
    """

    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _atomic_torch_save(obj, path: Path) -> None:
    """Write to a temp file, then rename. A crash mid-write can never leave a
    corrupt checkpoint behind, because os.replace is atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")    
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _checkpoint_path(configs: ModelConfig, step: int) -> Path:
    # Zero-padded so lexicographic order == step order.
    return Path(configs.checkpoint_dir) / f"{configs.model_name}_step{step:08d}.pt"


def list_checkpoints(configs: ModelConfig) -> list[Path]:
    """All checkpoints for this model, oldest first."""
    return sorted(Path(configs.checkpoint_dir).glob(f"{configs.model_name}_step*.pt"))


def find_latest_checkpoint(configs: ModelConfig) -> Path | None:
    ckpts = list_checkpoints(configs)
    return ckpts[-1] if ckpts else None


def _rotate_checkpoints(configs: ModelConfig) -> None:
    keep = configs.keep_last_n_checkpoints
    if keep is None or keep <= 0:
        return
    for old in list_checkpoints(configs)[:-keep]:
        old.unlink()

def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    model: nn.Module,
    configs: ModelConfig,
    optimizer: torch.optim.Optimizer,
    step: int,                      # number of COMPLETED steps (= next step to run)
    train_history: list[float],
    val_history: list[tuple[int, float]],
    running_train_loss: float,
    steps_since_log: int,
    train_loader: BatchLoader,
    scheduler=None,
    scaler=None,
) -> Path:
    """Save everything needed to continue training exactly where it stopped."""

    state = {
        "version": CHECKPOINT_VERSION,
        "step": step,
        "config": asdict(configs),

        # "model": model.state_dict(),
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,

        # Data order: the loader's private RNG decides which windows we see next.
        "train_loader": train_loader.state_dict(),

        # Global RNG (dropout etc., if the model ever uses it).
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            ),
        },

        # Logging state, so curves continue seamlessly.
        "train_history": list(train_history),
        "val_history": [list(item) for item in val_history],
        "running_train_loss": float(running_train_loss),
        "steps_since_log": int(steps_since_log),
    }

    path = _checkpoint_path(configs, step)
    _atomic_torch_save(state, path)
    _rotate_checkpoints(configs)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    configs: ModelConfig,
    optimizer: torch.optim.Optimizer,
    train_loader: BatchLoader,
    scheduler=None,
    scaler=None,
) -> dict:
    """
    Restore model, optimizer, loader and RNG state in place.
    Returns the checkpoint dict (use ckpt["step"], ckpt["train_history"], ...).

    `model` must already be on its final device, and `optimizer` must have been
    created from that model's parameters.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load on CPU: model/optimizer load_state_dict copy onto the right device,
    # and RNG states must be CPU ByteTensors anyway.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    if ckpt.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {ckpt.get('version')} "
            f"(expected {CHECKPOINT_VERSION})"
        )

    # Refuse to resume into a different architecture / data setup.
    saved_cfg, current_cfg = ckpt["config"], asdict(configs)
    mismatches = {
        key: (saved_cfg.get(key), current_cfg[key])
        for key in current_cfg
        if key not in _RESUME_IGNORED_CONFIG_KEYS
        and saved_cfg.get(key) != current_cfg[key]
    }
    if mismatches:
        details = "\n".join(
            f"  {k}: checkpoint={a!r} vs current={b!r}"
            for k, (a, b) in mismatches.items()
        )
        raise ValueError(f"Config differs from checkpoint:\n{details}")

    _unwrap(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    # load_state_dict also restores the *saved* learning rate. The current
    # config is the source of truth, so changing learning_rate between runs
    # works as expected. (Remove this if you use a scheduler that owns the LR.)
    if scheduler is None:
        for group in optimizer.param_groups:
            group["lr"] = configs.learning_rate

    if scheduler is not None:
        if ckpt["scheduler"] is None:
            raise ValueError(
                "Checkpoint was saved without an LR scheduler, so the schedule "
                "position can't be restored. Resume without a scheduler, or "
                "restart training."
            )
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])

    train_loader.load_state_dict(ckpt["train_loader"])

    torch.set_rng_state(ckpt["rng"]["torch_cpu"])
    cuda_states = ckpt["rng"]["torch_cuda"]
    if (
        cuda_states is not None
        and torch.cuda.is_available()
        and len(cuda_states) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(cuda_states)

    print(f"Resumed from {path} at step {ckpt['step']}")
    return ckpt


def _resolve_resume_path(configs: ModelConfig, resume_from) -> Path | None:
    if resume_from is None:
        return None
    if resume_from == "latest":
        latest = find_latest_checkpoint(configs)
        if latest is None:
            print(f"No checkpoints in {configs.checkpoint_dir}; starting from scratch.")
        return latest
    return Path(resume_from)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save_model(
    model: torch.nn.Module,
    configs: ModelConfig,
    device,
    training_steps: int,            # TOTAL target steps (not "additional" steps)
    eval_interval: int = 50,
    eval_batches: int = 20,
    seed: int = 42,
    log_interval: int = 10,
    resume_from: str | Path | None = None,   # None | "latest" | path to .pt
    # LR schedule (None -> constant LR). Keep these IDENTICAL across stages.
    schedule_total_steps: int | None = None,  # length of the whole planned run
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
):
    """
    Train for `training_steps` total steps, checkpointing every
    `configs.checkpoint_every_steps` (0 disables periodic checkpoints) and once
    more at the end. Pass resume_from="latest" (or a path) to continue.

    If `schedule_total_steps` is given, uses linear warmup + cosine decay over
    that many steps (see build_lr_scheduler).

    Returns: train_history (list[float]), val_history (list[(step, val_loss)])
    """

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configs.learning_rate,
    )

    scheduler = None
    if schedule_total_steps is not None:
        scheduler = build_lr_scheduler(
            optimizer, schedule_total_steps, warmup_steps, min_lr_ratio,
        )

    train_loader = BatchLoader(
        split="train",
        max_batch_size=configs.max_training_batch_size,
        max_seq_len=configs.max_context_window,
        device=device,
        seed=seed,
    )

    val_loader = BatchLoader(
        split="val",
        max_batch_size=configs.max_training_batch_size,
        max_seq_len=configs.max_context_window,
        device=device,
    )

    train_history: list[float] = []
    val_history: list[tuple[int, float]] = []
    start_step = 0

    # Accumulate on device so .item() does not happen every step.
    running_train_loss = torch.zeros((), device=device)
    steps_since_log = 0

    # ---- resume -----------------------------------------------------------
    ckpt_path = _resolve_resume_path(configs, resume_from)
    if ckpt_path is not None:
        ckpt = load_checkpoint(
            ckpt_path, model, configs, optimizer, train_loader,
            scheduler=scheduler,
        )
        start_step = ckpt["step"]
        train_history = list(ckpt["train_history"])
        val_history = [tuple(item) for item in ckpt["val_history"]]
        running_train_loss.fill_(ckpt["running_train_loss"])
        steps_since_log = ckpt["steps_since_log"]

        if start_step >= training_steps:
            print(
                f"Checkpoint is already at step {start_step} >= "
                f"training_steps={training_steps}. Nothing to do."
            )
            return train_history, val_history

    model.train()


    is_moe = configs.ffn is not None and configs.ffn.MoE_num_experts > 1

    # ---- main loop --------------------------------------------------------
    for step in range(start_step, training_steps):
        x, y = train_loader.get_batch()

        if is_moe:
            logits, aux = model(x, return_aux_loss=True)
            ce = _cross_entropy(logits, y)
            loss = ce + aux
        else:
            ce = _cross_entropy(logits := model(x), y)
            loss = ce


        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_train_loss += loss.detach()
        steps_since_log += 1

        is_last = step == training_steps - 1

        should_log = (
            (step + 1) % log_interval == 0
            or step == 0
            or is_last
        )

        if should_log:
            mean_train_loss = (running_train_loss / steps_since_log).item()
            train_history.append(mean_train_loss)
            running_train_loss.zero_()
            steps_since_log = 0

        if step % eval_interval == 0 or is_last:
            val_loss, val_ppl = evaluate(
                model,
                val_loader,
                max_batches=None if is_last else eval_batches,
            )
            val_history.append((step, val_loss))

            display_train_loss = train_history[-1] if train_history else loss.item()

            print(
                f"Step {step:04d} | "
                f"train(avg) {display_train_loss:.4f} | "
                f"val {val_loss:.4f} | "
                f"val ppl {val_ppl:.2f} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
            )

        # Periodic checkpoint. The final one is written after the loop.
        completed = step + 1
        if (
            configs.checkpoint_every_steps
            and completed % configs.checkpoint_every_steps == 0
            and not is_last
        ):
            save_checkpoint(
                model, configs, optimizer, completed,
                train_history, val_history,
                running_train_loss.item(), steps_since_log,
                train_loader,
                scheduler=scheduler,
            )

    # ---- finish -----------------------------------------------------------
    save_checkpoint(
        model, configs, optimizer, training_steps,
        train_history, val_history,
        running_train_loss.item(), steps_since_log,
        train_loader,
        scheduler=scheduler,
    )

    save_model_and_loss_logs(model, configs, val_history)

    return train_history, val_history


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: BatchLoader,
    max_batches: int | None = None,
):
    """
    Mean per-token cross-entropy over a deterministic validation pass.

    Returns:
        mean_loss_nats, perplexity
    """

    was_training = model.training
    model.eval()

    total_loss = None
    total_tokens = 0

    for x, y in loader.sequential_batches(max_batches):
        loss = _cross_entropy(model(x), y, reduction="sum")

        if total_loss is None:
            total_loss = loss.detach()
        else:
            total_loss += loss.detach()

        total_tokens += y.numel()

    if total_loss is None or total_tokens == 0:
        raise ValueError("Validation loader produced no tokens")

    mean_loss = (total_loss / total_tokens).item()
    perplexity = math.exp(mean_loss)

    model.train(was_training)

    return mean_loss, perplexity


# ---------------------------------------------------------------------------
# Final artifacts (weights-only file + loss log)
# ---------------------------------------------------------------------------

def save_model_and_loss_logs(
    model: nn.Module,
    configs: ModelConfig,
    val_history: list[tuple[int, float]],
):
    """Weights-only file for inference + 'step,loss' text log for plotting.
    (Resumable state lives in the checkpoints, not here.)"""

    Path(configs.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(configs.val_loss_history).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model weights to: {configs.save_path}")
    _atomic_torch_save(_unwrap(model).state_dict(), Path(configs.save_path))

    with open(configs.val_loss_history, "w", encoding="utf-8") as f:
        for step, loss in val_history:
            f.write(f"{step},{loss}\n")
    print(f"Loss history successfully saved to {configs.val_loss_history}")


def plot_loss_history(config: ModelConfig):

    steps, losses = [], []

    with open(config.val_loss_history, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if "," in line:                 # new format: step,loss
                step_str, loss_str = line.split(",")
                steps.append(int(step_str))
                losses.append(float(loss_str))
            else:                           # old format: loss only
                steps.append(i)
                losses.append(float(line))

    print(f"Loaded {len(losses)} evaluations from {config.val_loss_history}")

    random_loss = math.log(config.vocab_size)

    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, color='#00FFCC', label=f'{config.model_name} Validation Loss')
    plt.axhline(y=random_loss, color='red', linestyle='--',
                label=f'Theoretical Random Loss ln({config.vocab_size}) ≈ {random_loss:.4f}')
    plt.title("Loss Plot (TinyStories Dataset)")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss (Nats)")
    plt.grid(True, color='#333333')
    plt.legend()
    plt.show()


def inspect_weight_file(weight_file_path: str, print_weights: bool = False):

    weight_file_path = Path(weight_file_path)
    if not weight_file_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_file_path}")

    state_dict = torch.load(weight_file_path, map_location="cpu", weights_only=True)
    print(f"Loaded state_dict from {weight_file_path}")

    total_parameters = sum(param.numel() for param in state_dict.values())
    total_tensors = len(state_dict)

    print(f"Number of parameter tensors: {total_tensors}")
    print(f"Number of parameters: {total_parameters:,}")

    print("\n" + "=" * 80)
    print("PARAMETER SUMMARY")
    print("=" * 80)

    for name, param in state_dict.items():
        print(
            f"{name}: "
            f"shape={tuple(param.shape)}, "
            f"dtype={param.dtype}, "
            f"numel={param.numel():,}"
        )

    if print_weights:
        print("\n" + "=" * 80)
        print("PARAMETER TENSORS")
        print("=" * 80)

        for name, param in state_dict.items():
            print(f"\n{name}")
            print("-" * 80)
            print(param)