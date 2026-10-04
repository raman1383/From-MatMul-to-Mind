# get batch -> FWD -> BWD -> step. repeat

import math
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from einops import rearrange

from model_config import ModelConfig


class BatchLoader:
    def __init__(
            self, 
            split: str, 
            max_batch_size: int,
            max_seq_len: int ,
            device, 
            seed=None,
            #  train:bool, batch_size:int, max_seq_len:int, device
        ):

        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

    

        self.batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device 

        if split == "train":
            data_file = "../data/tokenized-2048-train-tinyStories-10Mb.txt"
        else:
            data_file = "../data/tokenized-2048-valid-tinyStories-1Mb.txt"


        self.tokens = self._load_tokens(data_file)


    def _load_tokens(self, path):
        path = Path(path)
        text = path.read_text(encoding="utf-8")

        # Assumes the file contains token IDs separated by whitespace.
        arr = np.fromstring(text, dtype=np.int64, sep=" ")

        if arr.size == 0:
            raise ValueError(f"No token IDs found in {path}")
        
        return torch.tensor(arr, dtype=torch.long)


    def get_batch(self):
        """Random windows (for training). Highest valid start is n_tokens - seq_len - 1."""

        tokens = self.tokens

        # Start positions must leave room for x and shifted y.
        max_start = len(tokens) - self.max_seq_len - 1
        starts = torch.randint(0, max_start + 1, (self.batch_size,))

        x = torch.stack([tokens[i : i + self.max_seq_len] for i in starts])
        y = torch.stack([tokens[i + 1 : i + 1 + self.max_seq_len] for i in starts])

        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        return x, y



    # def sequential_batches(self, max_batches=None):
    #     """
    #     Deterministic, non-overlapping windows covering the file in order (for validation):
    #     the same tokens are scored on every call, so val loss is comparable across runs.
    #     """
    #     n_windows = (self.n_tokens - 1) // self.seq_len
    #     starts = torch.arange(n_windows) * self.seq_len
    #     for b, i in enumerate(range(0, n_windows, self.batch_size)):
    #         if max_batches is not None and b >= max_batches:
    #             break
    #         yield self._gather(starts[i:i + self.batch_size])




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



def train_and_save_model(
        model:torch.nn.Module, 
        configs:ModelConfig, 
        device, 
        training_steps:int,

        eval_interval: int = 50, 
        eval_batches: int = 20, 
        seed: int = 42,
    ):

    torch.manual_seed(seed)
    train_history, val_history = [], []      # val_history: (step, loss)


    # Instantiate Model & optimizer
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=configs.learning_rate)

    # trainer_loader = BatchLoader(
    #     train=True,
    #     batch_size=configs.max_training_batch_size,
    #     max_seq_len=configs.max_context_window,
    #     device=device,
    # )

    train_loader = BatchLoader("train", configs.max_training_batch_size, configs.max_context_window, device, seed=seed)
    val_loader   = BatchLoader("val",   configs.max_training_batch_size, configs.max_context_window, device)


    
    model.train()
    for step in range(training_steps):

        x, y = train_loader.get_batch()

        loss = _cross_entropy(model(x), y)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Log metrics
        train_history.append(loss.item())

        # if step % 5 == 0 or step == training_steps - 1:
        #     print(f"Step {step:04d} | Batch Loss: {loss.item():.4f} nats")

        if step % eval_interval == 0 or step == training_steps - 1:
            # cheap partial eval during training, full pass at the very end
            is_last = step == training_steps - 1
            # val_loss, val_ppl = evaluate(model, val_loader, None if is_last else eval_batches)
            # val_history.append((step, val_loss))
            print(f"Step {step:04d} | train(batch) {loss.item():.4f} | "
                #   f"val {val_loss:.4f} | val ppl {val_ppl:.1f}"
                  )

    save_model_and_loss_logs(model, configs, train_history)


    # val_path = Path(configs.val_loss_history)

    # with open(val_path, "w", encoding="utf-8") as f:
    #     for step, v in val_history:
    #         f.write(f"{step} {v}\n")
    # print(f"Validation history saved to {val_path}")


    # return train_history, val_history





# @torch.no_grad()
# def evaluate(model: nn.Module, loader: BatchLoader, max_batches=None):
#     """
#     Mean per-token cross-entropy over a deterministic pass of the validation set.
#     Sums the loss and divides by the token count, so a smaller last batch is weighted correctly.
#     Returns (val_loss_nats, perplexity).
#     """
#     was_training = model.training
#     model.eval()
 
#     total_loss, total_tokens = 0.0, 0
#     for x, y in loader.sequential_batches(max_batches):
#         total_loss += _cross_entropy(model(x), y, reduction="sum").item()
#         total_tokens += y.numel()
 
#     model.train(was_training)
 
#     mean_loss = total_loss / total_tokens
#     return mean_loss, math.exp(mean_loss)




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


    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='#00FFCC', label=f'{config.model_name} Training Loss')
    plt.axhline(y=torch.log(torch.tensor(config.vocab_size)).item(), color='red', linestyle='--', 
                label=f'Theoretical Random Loss ln({config.vocab_size}) ≈ {math.log(config.vocab_size):.4f})')
    plt.title("Loss Plot (TinyStories Dataset)")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss (Nats)")
    plt.style.use('dark_background')
    plt.grid(True, color='#333333')
    plt.legend()
    plt.show()


def inspect_weight_file(weight_file_path: str, print_weights: bool=False):

    weight_file_path = Path(weight_file_path)
    if not weight_file_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_file_path}")

    state_dict = torch.load(weight_file_path, map_location="cpu")
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
            f"requires_grad={param.requires_grad}, "
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