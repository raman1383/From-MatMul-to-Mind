import os
import torch
from typing import Optional
import torch.nn.functional as F
from einops import rearrange, pack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from tokenizerModule import ID_mapper



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


# ----------------------------------------------------------------------


def sample_next_token_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Sample one next token from logits.

    Input:
        logits: [B, Vocab_Size]

    Output:
        next_token: [B, 1]
    """

    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [B, V], got {tuple(logits.shape)}"
        )

    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    if temperature == 0.0:
        # Greedy decoding
        return torch.argmax(logits, dim=-1, keepdim=True)

    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive or None")
        top_k = min(top_k, logits.shape[-1])

    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1] or None")

    # 1. Temperature scaling
    logits = logits / temperature

    # 2. Top-k filtering
    if top_k is not None:
        kth_value = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_value, float("-inf"))

    # 3. Top-p filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )

        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep the first token that crosses the threshold.
        sorted_indices_to_remove[..., 1:] = (
            sorted_indices_to_remove[..., :-1].clone()
        )
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )

        logits = logits.masked_fill(
            indices_to_remove,
            float("-inf"),
        )

    # 4. Convert to probabilities and sample
    probabilities = F.softmax(logits, dim=-1)

    return torch.multinomial(
        probabilities,
        num_samples=1,
    )

# ----------------------------------------------------------------------



@torch.inference_mode()
def batched_simple_inference(
    model_scaffold: torch.nn.Module,
    configs,
    num_gen_steps: int,
    prompt_strs: list[str],
    device,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
) -> list[str]:
    """
    Generate `num_gen_steps` tokens for every prompt.

    All prompt strings must tokenize to the same sequence length.

    Input:
        prompt_strs: list[str]

    Returns:
        list[str]
        Each output string contains its original prompt plus
        `num_get_steps` generated tokens.
    """

    if num_gen_steps < 0:
        raise ValueError("num_get_steps must be non-negative")

    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    if len(prompt_strs) == 0:
        raise ValueError("prompt_strs must contain at least one prompt")

    # ------------------------------------------------------------
    # 1. Encode every prompt
    # ------------------------------------------------------------

    tensorized_prompts = [
        torch.tensor(
            ID_mapper.encode(prompt),
            dtype=torch.long,
        )
        for prompt in prompt_strs
    ]

    # All prompts must have identical token lengths
    prompt_lengths = [tensor.shape[0] for tensor in tensorized_prompts]

    if len(set(prompt_lengths)) != 1:
        raise ValueError(
            f"All prompts must have the same token length, "
            f"got lengths: {prompt_lengths}"
        )

    # [B, seq_len]
    prompts = torch.stack(tensorized_prompts).to(device)

    # print("prompts.shape ->", prompts.shape)

    # ------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------

    loaded_model = load_model_weights(
        model_scaffold,
        configs["save_path"],
        device,
    )

    loaded_model.eval()

    # This tensor will grow autoregressively:
    #
    # [B, prompt_len]
    #      ↓
    # [B, prompt_len + 1]
    #      ↓
    # [B, prompt_len + 2]
    #      ...
    generated_sequences = prompts

    # ------------------------------------------------------------
    # 3. Autoregressive generation
    # ------------------------------------------------------------

    for step in range(num_gen_steps):

        # print(
        #     f"[INFO] Generation step "
        #     f"{step + 1}/{num_get_steps}"
        # )

        # Keep only the portion that fits in the model's
        # training context window.
        context_window_limited_sequences = (
            generated_sequences[:, -configs["max_context_window"]:]
        )

        # print(
        #     "context_window_limited_sequences.shape ->",
        #     context_window_limited_sequences.shape,
        # )

        # --------------------------------------------------------
        # [B, seq_len] -> [B, seq_len, vocab_size]
        # --------------------------------------------------------

        model_output_logits = loaded_model(
            context_window_limited_sequences
        )

        # print(
        #     "model_output_logits.shape ->",
        #     model_output_logits.shape,
        # )

        # We only care about the logits for the final position.
        #
        # [B, seq_len, V]
        #        ↓
        # [B, V]
        last_token_logits = model_output_logits[:, -1, :]

        # print(
        #     "last_token_logits.shape ->",
        #     last_token_logits.shape,
        # )

        # --------------------------------------------------------
        # Sample one token PER prompt
        #
        # [B, V] -> [B, 1]
        # --------------------------------------------------------

        next_token = sample_next_token_from_logits(
            logits=last_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # print(
        #     "next_token.shape ->",
        #     next_token.shape,
        # )

        # --------------------------------------------------------
        # Append one generated token to every sequence
        #
        # [B, S] + [B, 1]
        #        ↓
        # [B, S + 1]
        # --------------------------------------------------------

        generated_sequences = torch.cat(
            [
                generated_sequences,
                next_token,
            ],
            dim=-1,
        )

        # print(
        #     "generated_sequences.shape ->",
        #     generated_sequences.shape,
        # )

    # ------------------------------------------------------------
    # 4. Convert each generated token sequence back to a string
    # ------------------------------------------------------------

    decoded_list = []

    for sequence in generated_sequences:
        token_ids = sequence.tolist()

        decoded_list.append(
            ID_mapper.decode(token_ids)
        )

    return decoded_list



    

# ----------------------------------------------------------------------

# @dataclass
class KV_cache:
    """
    Static pre-allocated KV cache with sliding-window support.
    Fully compatible with MHA / GQA / MQA / SHA.
    
    [num_layers, batch, num_kv_heads, max_context_window, head_dim]

    The attention module is responsible for expanding KV heads
    to match num_q_heads (repeat_interleave) when necessary.
    """

    def __init__(
        self,
        config: dict,
        inference_batch_size:int = 1,
        device: torch.device = 'cuda',
        dtype: torch.dtype = torch.float16,

    ):
        self.max_context_window   = config["max_context_window"]
        self.inference_batch_size = inference_batch_size

        self.head_dim     = config["embed_dim"] // config["num_q_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.num_layers   = config["num_layers"]

        self.device = device
        self.dtype  = dtype


        shape = (self.num_layers, 
                 self.inference_batch_size, 
                 self.num_kv_heads, 
                 self.max_context_window, 
                 self.head_dim
        )

        self.keys   = torch.zeros(shape, device=device, dtype=dtype)
        self.values = torch.zeros(shape, device=device, dtype=dtype)

        self.position = 0  # absolute position counter(grows forever)
        self.seq_len  = 0  # how many slots are currently filled

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,         # [B, num_kv_heads, q_len, head_dim] (RoPE already applied)
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        q_len = key.shape[2]
        
        if layer_idx == 0 and self.seq_len + q_len > self.max_context_window:
            self._shift(q_len)

        start, end = self.seq_len, self.seq_len + q_len
        self.keys  [layer_idx, :, :, start:end, :] = key
        self.values[layer_idx, :, :, start:end, :] = value

        if layer_idx == self.num_layers - 1:      # advance once per forward pass
            self.seq_len   = end
            self.position += q_len

        return (self.keys  [layer_idx, :, :, :end, :],
                self.values[layer_idx, :, :, :end, :])


    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        usable = min(self.seq_len, self.max_context_window)
        start  = self.seq_len - usable

        return (
            self.keys  [layer_idx, :, :, start:self.seq_len, :],
            self.values[layer_idx, :, :, start:self.seq_len, :],
        )


    def _shift(self, incoming: int):
        keep = max(0, self.max_context_window - incoming)
        if keep > 0:
            self.keys  [..., :keep, :] = self.keys  [..., self.seq_len-keep:self.seq_len, :].clone()
            self.values[..., :keep, :] = self.values[..., self.seq_len-keep:self.seq_len, :].clone()
        self.seq_len = keep


    def reset(self):
        # No need to zero memory — just rewind the pointer.
        # Stale data past max_context_window gets over-written.
        self.seq_len  = 0
        self.position = 0

    @property
    def current_abs_pos(self) -> int:
        return self.position



# disables gradient recording and removes additional 
# autograd overhead such as version-counter/view tracking.
@torch.inference_mode()
def advanced_inference(
    model_scaffold: torch.nn.Module,
    configs: Dict[str, Any],
    num_gen_steps: int,
    prompt_strs: list[str],
    device: Union[str, torch.device],
    temperature: float = 0.6,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    eos_token_id: Optional[int] = None,
) -> list[str]:

    
    """

    Autoregressive generation split into two distinct stages:
        1. Prefill Stage: Process all prompt tokens at once, populating the KV cache.
        2. Decode Stage: Process tokens step-by-step (1 token input per step).
    
    """

    # pre-allocate a max_context_len sized pair torch.tensor, 


    tensorized_prompts = [
        torch.tensor(
            ID_mapper.encode(prompt),
            dtype=torch.long,
        )
        for prompt in prompt_strs
    ]

    prompts = torch.stack(tensorized_prompts).to(device) # [batch_size, seq_len]

    batch_size, seq_len = prompts.shape[0], prompts.shape[1]


    model = load_model_weights(
        model_scaffold,
        configs["save_path"],
        device,
    )

    model.eval()

    cache = KV_cache(configs, device=device, inference_batch_size=batch_size, dtype=next(model.parameters()).dtype)

    # Prefill
    logits = model(prompts, cache=cache)
    next_token = sample_next_token_from_logits(logits[:, -1, :], temperature, top_k, top_p)
    generated = [next_token]


    # Decode – can run for any number of steps
    for _ in range(num_gen_steps - 1):

        logits = model(next_token, cache=cache)
        next_token = sample_next_token_from_logits(logits[:, -1, :], temperature, top_k)
        generated.append(next_token)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break


    cache.reset()

    generated = torch.cat(generated, dim=1)          # [B, num_gen_steps]
    decoded_list = [ID_mapper.decode(row.tolist()) for row in generated]
    
    return decoded_list
