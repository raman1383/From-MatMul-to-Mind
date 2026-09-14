# get batch -> FWD -> BWD -> step. repeat

import math
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from einops import rearrange


def train_and_save_model(model, configs:dict, device, training_steps:int):

    loss_history = []

    # Instantiate Model & optimizer
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=configs["learning_rate"])

    trainer_loader = BatchLoader(
        train=True,
        batch_size=configs["batch_size"],
        max_seq_len=configs["max_seq_len"],
        device=device,
    )

    
    model.train()
    for step in range(training_steps):

        x, y = trainer_loader.get_batch()

        # Forward pass though the model
        logits = model(x)

        logits_flat = rearrange(logits, "batch_size seq_len vocab_size -> (batch_size seq_len) vocab_size")
        targets_flat = rearrange(y, "batch_size seq_len -> (batch_size seq_len)")

        loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Log metrics
        loss_history.append(loss.item())
        if step % 5 == 0 or step == training_steps - 1:
            print(f"Step {step:04d} | Batch Loss: {loss.item():.4f} nats")


    save_model_and_loss_logs(model, configs, loss_history)


class BatchLoader:
    def __init__(self, train:bool, batch_size:int, max_seq_len:int, device):

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device 

        if train == True:
            data_file = "./data/tokenized-2048-train-tinyStories-10Mb.txt"
        else:
            data_file = "./data/tokenized-2048-valid-tinyStories-1Mb.txt"


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
        tokens = self.tokens

        # Start positions must leave room for x and shifted y.
        max_start = len(tokens) - self.max_seq_len - 1
        starts = torch.randint(0, max_start + 1, (self.batch_size,))

        x = torch.stack([tokens[i : i + self.max_seq_len] for i in starts])
        y = torch.stack([tokens[i + 1 : i + 1 + self.max_seq_len] for i in starts])

        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        return x, y


def save_model_and_loss_logs(model, configs, loss_logs):

    Path(configs["save_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(configs["loss_history"]).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model weights to: {configs["save_path"]}")
    torch.save(model.state_dict(), configs["save_path"])


    """Saves a list of loss floats to a text file, one per line."""
    with open(configs["loss_history"], "w", encoding="utf-8") as f:
        for loss in loss_logs:
            f.write(f"{loss}\n")
    print(f"Loss history successfully saved to {configs["loss_history"]}")


def plot_loss_history(config):

    loss_history = []

    with open(config["loss_history"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                loss_history.append(float(line))

    print(f"Loaded {len(loss_history)} steps from {config["loss_history"]}")


    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='#00FFCC', label='Bigram Training Loss')
    plt.axhline(y=torch.log(torch.tensor(config["vocab_size"])).item(), color='red', linestyle='--', 
                label=f'Theoretical Random Loss ln({config["vocab_size"]}) ≈ {math.log(config["vocab_size"]):.4f})')
    plt.title("Level 1 Loss Reduction Chronicle (TinyStories Dataset)")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss (Nats)")
    plt.style.use('dark_background')
    plt.grid(True, color='#333333')
    plt.legend()
    # plt.savefig(config["loss_history_pic"], dpi=300)
    plt.show()



# loader = BatchLoader(train = True, batch_size=4, max_seq_len=12, device='cuda')
# x, y = loader.get_batch()
# print(x, y)
