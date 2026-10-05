import math
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from einops import rearrange

from data_loader import BatchLoader
from model_config import ModelConfig




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



# TODO: instead of jus logging metrics, save the model checkpoint too, must be able to load and restart training from it.
def train_and_save_model(
    model: torch.nn.Module,
    configs: ModelConfig,
    device,
    training_steps: int,
    eval_interval: int = 50,
    eval_batches: int = 20,
    seed: int = 42,
    log_interval: int = 10,
):

    torch.manual_seed(seed)
    
    train_history = []
    val_history = []

    # Instantiate model & optimizer
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configs.learning_rate,
    )

    train_loader = BatchLoader(
        split="train",
        max_batch_size=configs.max_training_batch_size,
        max_seq_len=configs.max_context_window,
        device=device,
        seed=seed,
    )

    # Fixed seed here is not necessary because validation is deterministic,
    # but leaving it explicit makes the loader configuration unambiguous.
    val_loader = BatchLoader(
        split="val",
        max_batch_size=configs.max_training_batch_size,
        max_seq_len=configs.max_context_window,
        device=device,
    )

    model.train()

    # Accumulate on device so .item() does not happen every step.
    running_train_loss = torch.zeros((), device=device)
    steps_since_log = 0


    for step in range(training_steps):
        x, y = train_loader.get_batch()

        logits = model(x)
        loss = _cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.detach()
        steps_since_log += 1

        should_log = (
            (step + 1) % log_interval == 0
            or step == 0
            or step == training_steps - 1
        )


        if should_log:
            mean_train_loss = (
                running_train_loss / steps_since_log
            ).item()

            train_history.append(mean_train_loss)

            running_train_loss.zero_()
            steps_since_log = 0



        if step % eval_interval == 0 or step == training_steps - 1:
            is_last = step == training_steps - 1

            val_loss, val_ppl = evaluate(
                model,
                val_loader,
                max_batches=None if is_last else eval_batches,
            )

            val_history.append((step, val_loss))

            # Use the most recent logged training loss.
            if train_history:
                display_train_loss = train_history[-1]
            else:
                display_train_loss = loss.item()

            print(
                f"Step {step:04d} | "
                f"train(avg) {display_train_loss:.4f} | "
                f"val {val_loss:.4f} | "
                f"val ppl {val_ppl:.2f}"
            )
            
    save_model_and_loss_logs(
        model,
        configs,
        train_history,
    )

    return train_history, val_history





@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: BatchLoader,
    max_batches: int | None = None,
):
    """
    Mean per-token cross-entropy over a deterministic validation pass.

    The same validation target tokens are evaluated in the same order
    every time this function is called.

    Returns:
        mean_loss_nats, perplexity
    """

    was_training = model.training
    model.eval()

    total_loss = None
    total_tokens = 0

    for x, y in loader.sequential_batches(max_batches):
        loss = _cross_entropy(
            model(x),
            y,
            reduction="sum",
        )

        # Keep accumulation on the same device as the model.
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



# TODO: Checkpoints can't resume. Save the config, optimizer state, and step alongside the weights.
def save_model_and_loss_logs(model, configs:ModelConfig, loss_logs):

    Path(configs.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(configs.val_loss_history).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model weights to: {configs.save_path}")
    torch.save(model.state_dict(), configs.save_path)


    """Saves a list of loss floats to a text file, one per line."""
    with open(configs.val_loss_history, "w", encoding="utf-8") as f:
        for loss in loss_logs:
            f.write(f"{loss}\n")
    print(f"Loss history successfully saved to {configs.val_loss_history}")


def plot_loss_history(config:ModelConfig):

    loss_history = []

    with open(config.val_loss_history, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                loss_history.append(float(line))

    print(f"Loaded {len(loss_history)} steps from {config.val_loss_history}")


    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='#00FFCC', label=f'{config.model_name} Training Loss')
    plt.axhline(y=torch.log(torch.tensor(config.vocab_size)).item(), color='red', linestyle='--', 
                label=f'Theoretical Random Loss ln({config.vocab_size}) ≈ {math.log(config.vocab_size):.4f})')
    plt.title("Loss Plot (TinyStories Dataset)")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss (Nats)")
    plt.grid(True, color='#333333')
    plt.legend()
    plt.show()


def inspect_weight_file(weight_file_path: str, print_weights: bool=False):

    weight_file_path = Path(weight_file_path)
    if not weight_file_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_file_path}")

    state_dict = torch.load(weight_file_path, map_location="cpu", weights_only=True)
    print(f"Loaded state_dict from {weight_file_path}")

    # Number of individual scalar parameters
    total_parameters = sum(param.numel() for param in state_dict.values())

    # Number of parameter tensors
    total_tensors = len(state_dict)

    print(f"Number of parameter tensors: {total_tensors}")
    print(f"Number of parameters: {total_parameters:,}")


    # PARAMETER SUMMARY

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


    # PARAMETER TENSORS

    if print_weights:
        print("\n" + "=" * 80)
        print("PARAMETER TENSORS")
        print("=" * 80)

        for name, param in state_dict.items():
            print(f"\n{name}")
            print("-" * 80)
            print(param)