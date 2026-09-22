import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange

from tokenizerModule import ID_mapper

@dataclass
class KVCache:
    caches: List[Dict[str, Optional[torch.Tensor]]] = field(default_factory=list)
    seq_len: int = 0
    # After truncation, this tells the model what absolute position
    # the *first* token currently in the cache corresponds to.
    start_offset: int = 0

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        while len(self.caches) <= layer_idx:
            self.caches.append({"k": None, "v": None})

        if self.caches[layer_idx]["k"] is None:
            self.caches[layer_idx]["k"] = key
            self.caches[layer_idx]["v"] = value
        else:
            self.caches[layer_idx]["k"] = torch.cat(
                [self.caches[layer_idx]["k"], key], dim=-2
            )
            self.caches[layer_idx]["v"] = torch.cat(
                [self.caches[layer_idx]["v"], value], dim=-2
            )
        return self.caches[layer_idx]["k"], self.caches[layer_idx]["v"]

    def truncate(self, max_len: int) -> None:
        """Keep only the last max_len tokens and adjust the position offset."""
        if self.seq_len <= max_len:
            return

        dropped = self.seq_len - max_len
        self.start_offset += dropped          # ← critical line

        for cache in self.caches:
            if cache["k"] is not None:
                cache["k"] = cache["k"][:, :, -max_len:, :].contiguous()
                cache["v"] = cache["v"][:, :, -max_len:, :].contiguous()

        self.seq_len = max_len

    def reset(self):
        self.caches = []
        self.seq_len = 0
        self.start_offset = 0

def load_model_weights(
    model_scaffold: torch.nn.Module,
    save_path: str,
    device: torch.device,
) -> Optional[torch.nn.Module]:
    if not os.path.isfile(save_path):
        print(f"[ERROR] Weight file not found: {save_path}")
        return None

    try:
        state_dict = torch.load(save_path, map_location=device, weights_only=True)
        model_scaffold.load_state_dict(state_dict)
        model_scaffold = model_scaffold.to(device)
        model_scaffold.eval()

        n_params = sum(p.numel() for p in model_scaffold.parameters())
        print(f"[INFO] Loaded weights from {save_path}  ({n_params:,} parameters)")
        return model_scaffold
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        return None


kv_cache:list[tuple[torch.tensor, torch.tensor]] = []


@torch.inference_mode()
def advanced_inference(
    model_scaffold: torch.nn.Module,
    configs: Dict[str, Any],
    max_new_tokens: int,
    prompt: str,
    device: Union[str, torch.device],
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
) -> str:
    max_seq_len = configs["max_seq_len"]

    prompt_ids = ID_mapper.encode(prompt)
    if not prompt_ids:
        raise ValueError("[ERR] Prompt produced no tokens.")

    model = load_model_weights(model_scaffold, configs["save_path"], device)
    if model is None:
        raise RuntimeError("Failed to load model checkpoint.")

    kv_cache = KVCache()
    full_seq = list(prompt_ids)

    # ---------- PREFILL ----------
    prefill_ids = full_seq[-max_seq_len:]
    last_logits = prefill(model, prefill_ids, kv_cache, device)
    kv_cache.seq_len = len(prefill_ids)          # ≤ max_seq_len
    # start_offset is still 0

    next_token = sample(last_logits, temperature, top_k, top_p).item()

    # ---------- DECODE ----------
    for _ in range(max_new_tokens):
        if eos_token_id is not None and next_token == eos_token_id:
            break

        # 1. Append the newly generated token
        full_seq.append(next_token)

        # 2. Truncate FIRST (before calling the model)
        if len(full_seq) > max_seq_len:
            full_seq = full_seq[-max_seq_len:]
            kv_cache.truncate(max_seq_len)        # updates seq_len and start_offset

        # 3. Now the cache is guaranteed to have space / correct offset
        logits = decode(model, next_token, kv_cache, device)
        next_token = sample(logits, temperature, top_k, top_p).item()

    return ID_mapper.decode(full_seq)

def prefill(model, prompt_tokens, kv_cache, device):
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    logits = model(
        input_ids,
        kv_cache=kv_cache,
        start_pos=kv_cache.start_offset,          # normally 0
    )
    return logits[:, -1, :]


def decode(model, next_token_id, kv_cache, device):
    input_id = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

    # After possible truncation, seq_len is already the length *before*
    # adding the new token. So the new token’s absolute position is:
    start_pos = kv_cache.start_offset + kv_cache.seq_len

    logits = model(
        input_id,
        kv_cache=kv_cache,
        start_pos=start_pos,
    )
    kv_cache.seq_len += 1
    return logits[:, -1, :]


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    if logits.dim() > 1:
        logits = logits.squeeze(0)

    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-6)

    # Top-k
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # Top-p (nucleus)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        logits[sorted_idx[mask]] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)




@torch.inference_mode()
def simple_inference(model_scaffold:torch.nn.Module, 
                               configs, 
                               num_gen_steps: int, 
                               prompt: str, 
                               device, 
                               temperature: float
                               ) -> str:

    print(f"[INFO] Starting inference with prompt: {prompt}")

    if num_gen_steps < 0:
        raise ValueError("[ERR] num_gen_steps must be non-negative")

    if temperature < 0:
        raise ValueError("[ERR] temperature must be non-negative")

    prompt_to_IDs = ID_mapper.encode(prompt)
    if len(prompt_to_IDs) == 0:
        raise ValueError("[ERR] Prompt produced no tokens. Provide a non-empty prompt.")

    loaded_model = load_model_weights(model_scaffold, configs["save_path"], device)


    # a list for progressively appending our generated predictions to.
    # full_seq = prompt + generated tokens
    full_seq = list(prompt_to_IDs)

    loaded_model.eval()
    with torch.no_grad():
        for step in range(num_gen_steps):

            print(f"full_seq before {step+1}/{num_gen_steps}:")
            print(full_seq)

            # clip long sequences to fit training context window
            context_window_limited_seq = full_seq[-configs["max_seq_len"] :]

            # convert to torch.tensor
            # (batch_size=1, seq_len=len(context_window_limited_seq))
            context_window_limited_seq_tensor = torch.tensor(
                [context_window_limited_seq], 
                device=device, 
                dtype=torch.long
            )
            print("context_window_limited_seq_tensor.shape-> ")
            print(context_window_limited_seq_tensor.shape)
            print(f"[INFO] Step {step+1}/{num_gen_steps}: Context window length: {context_window_limited_seq_tensor.shape[1]}")

            # (batch_size=1, seq_len=len(context_window_limited_seq), vocab_size)
            model_output_logits = loaded_model(context_window_limited_seq_tensor)
            print("model_output_logits.shape->")
            print(model_output_logits.shape)

            # extract the logits for ONLY the final token position in the sequence
            last_token_logits = rearrange(model_output_logits[:, -1, :], "1 vocab_size -> vocab_size")
            print("last_token_logits.shape->")
            print(last_token_logits.shape)

            # temperature controls the sharpness of the distribution
            tempered_logits = last_token_logits / max(temperature, 1e-6)

            # convert to probability distribution
            probabilities = torch.softmax(tempered_logits, dim=-1)
            print("probabilities.shape-> ")
            print(probabilities.shape)

            print("probabilities-> ")
            print(probabilities)

            # multinomial sampling
            next_token_tensor = torch.multinomial(probabilities, num_samples=1)
            print("next_token_tensor.shape-> ")
            print(next_token_tensor.shape)
            print("next_token_tensor-> ")
            print(next_token_tensor)
            next_token = next_token_tensor.item()

            # Append predicted token to sequence for the next autoregressive loop step
            full_seq.append(next_token)
            print("full_seq.append after append: ")
            print(full_seq)
            print("-"*80)


    return ID_mapper.decode(full_seq)




