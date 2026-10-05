import math
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from einops import rearrange

from model_config import ModelConfig



def convert_txt_to_bin(txt_path: str | Path, bin_path: str | Path) -> None:
    """
    One-time conversion:

        whitespace-separated token IDs
        -> 
        uint16 binary file

    Vocab size 2048 easily fits in uint16.
    """

    txt_path = Path(txt_path)
    bin_path = Path(bin_path)

    # np.fromfile replaces the deprecated np.fromstring usage.
    tokens = np.fromfile(txt_path, dtype=np.int64, sep=" ")

    if tokens.size == 0:
        raise ValueError(f"No token IDs found in {txt_path}")

    if tokens.min() < 0:
        raise ValueError(f"Negative token ID found in {txt_path}")

    if tokens.max() > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Token ID {tokens.max()} does not fit in uint16"
        )

    tokens = tokens.astype(np.uint16)

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(bin_path)

    print(
        f"Converted {txt_path} -> {bin_path} "
        f"({tokens.size:,} tokens)"
    )



# TODO: np.fromstring is deprecated. Store tokens as a uint16 .bin and use np.memmap. 
# Vocab 2048 fits easily, and you can build the batch with a single index tensor 
# instead of a Python list comprehension.
class BatchLoader:
    def __init__(
        self,
        split: str,
        max_batch_size: int,
        max_seq_len: int,
        device,
        seed: int | None = None,
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")

        
        self.split = split
        self.batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device



        if split == "train":
            data_file = "../data/tokenized-2048-train-tinyStories-10Mb.bin"      
        else:
            data_file = "../data/tokenized-2048-valid-tinyStories-1Mb.bin"


        self.tokens = self._load_tokens(data_file)



        if len(self.tokens) < self.max_seq_len + 1:
            raise ValueError(
                f"{data_file} contains only {len(self.tokens)} tokens, "
                f"but max_seq_len={self.max_seq_len} requires at least "
                f"{self.max_seq_len + 1}."
            )

        # Independent RNG for this loader.
        # This avoids changing the global PyTorch RNG state.
        self.rng = torch.Generator(device="cpu")

        if seed is None:
            self.rng.seed()
        else:
            self.rng.manual_seed(seed)

        # Reused for every batch.
        # Shape: [1, max_seq_len + 1]
        self.offsets = torch.arange(
            self.max_seq_len + 1,
            dtype=torch.long,
        ).unsqueeze(0)



    @staticmethod
    def _load_tokens(path: str | Path) -> np.memmap:
        
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Token file not found: {path}")

        if path.stat().st_size == 0:
            raise ValueError(f"Token file is empty: {path}")

        if path.stat().st_size % np.dtype(np.uint16).itemsize != 0:
            raise ValueError(
                f"Invalid uint16 .bin file size: {path}"
            )

        return np.memmap(
            path,
            dtype=np.uint16,
            mode="r",
        )


    def get_batch(self):
        """
        Random windows for training.

        Sampling is random, but deterministic when `seed` is supplied.
        """

        max_start = len(self.tokens) - self.max_seq_len - 1

        starts = torch.randint(
            low=0,
            high=max_start + 1,
            size=(self.batch_size,),
            generator=self.rng,
            device="cpu",
        )

        return self._make_batch(starts)


    def _make_batch(self, starts: torch.Tensor):
        """
        Build a batch from starting positions.

        starts:
            [batch_size]

        window_indices:
            [batch_size, max_seq_len + 1]

        The extra token allows us to construct:
            x = tokens[t : t + seq_len]
            y = tokens[t + 1 : t + 1 + seq_len]

        One indexed read gives both x and y.
        """

        starts = starts.to(dtype=torch.long)

        # [batch_size, 1] + [1, seq_len + 1]
        window_indices = starts.unsqueeze(1) + self.offsets

        # NumPy memmap advanced indexing.
        windows_uint16 = self.tokens[window_indices.numpy()]

        # Convert to torch and widen token IDs to int64 for embeddings/loss.
        windows = torch.from_numpy(windows_uint16).to(dtype=torch.long)

        x = windows[:, :-1]
        y = windows[:, 1:]

        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        return x, y



    def sequential_batches(self, max_batches: int | None = None):
        """
        Deterministic validation pass.

        Target tokens are scored in exactly the same order on every call.

        Starts are:
            0, seq_len, 2*seq_len, ...

        Therefore the target ranges are:

            [1, ..., seq_len]
            [seq_len+1, ..., 2*seq_len]
            [2*seq_len+1, ..., 3*seq_len]
            ...

        The final batch may contain fewer than batch_size sequences.
        """

        if max_batches is not None and max_batches < 0:
            raise ValueError("max_batches must be >= 0 or None")

        max_start = len(self.tokens) - self.max_seq_len - 1

        
        # Every validation window has a distinct set of target tokens.
        starts = torch.arange(
            0,
            max_start + 1,
            self.max_seq_len,
            dtype=torch.long,
        )

        num_batches = 0

        for batch_start in range(0, len(starts), self.batch_size):
            if max_batches is not None and num_batches >= max_batches:
                break

            batch_starts = starts[
                batch_start : batch_start + self.batch_size
            ]

            yield self._make_batch(batch_starts)

            num_batches += 1



