from collections import deque
import os
import torch
from pathlib import Path
from typing import Optional
import torch.nn.functional as F
from einops import rearrange, pack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from model_config import ModelConfig
from tokenizerModule import ID_mapper
from kv_cache import KV_cache


def load_model_weights(model: torch.nn.Module, save_path: str, device) -> torch.nn.Module:
    """Load a checkpoint into `model` and return it in eval mode."""
    path = Path(save_path)
 
    if not path.is_file():
        raise FileNotFoundError(f"[ERR] Weight file not found: {path.resolve()}")
 
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"[ERR] Could not read checkpoint {path}: {e}") from e
 
    # Raises RuntimeError listing missing / unexpected keys or shape mismatches.
    # (Checkpoints from before the fused-QKV / tied-embedding change won't load: retrain.)
    model.load_state_dict(state_dict)
 
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Loaded weights from {path} ({n_params:,} parameters)")
    return model

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
    configs:ModelConfig,
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
        configs.save_path,
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
            generated_sequences[:, -configs.max_context_window:]
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






@dataclass
class _Request:
    index: int
    prompt_ids: list # FULL prompt, never truncated
    max_new_tokens: int
    n_prefilled: int = 0 # prompt tokens already in the cache
    generated: list = field(default_factory=list)
    finish_reason: str = ""


@dataclass
class Completion:
    text: str            # prompt + generated
    completion: str      # generated only
    finish_reason: str   # "eos" | "length"



@torch.inference_mode()
def advanced_inference(

    model,
    configs: ModelConfig,
    prompt_strs: list[str],
    
    device,
    max_new_tokens: int = 64,

    max_KV_lanes: int = 4,               # max sequences in flight (cache lanes)
    chunk_size: int = 32,             # prompt tokens prefilled per iteration


    temperature: float = 0.6,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    eos_token_id: Optional[int] = None,

) -> list[Completion]:

    
    """

    Iteration-level scheduling with chunked prefill and a ring-buffer KV cache.
    Prompts and generations may be arbitrarily long; the cache stays at
    max_context_window tokens per lane. Returns one Completion per prompt, in input order.
    
    """

    model.eval()

    num_lanes = min(max_KV_lanes, len(prompt_strs))

    cache = KV_cache(
        configs, 
        num_lanes=num_lanes, 
        device=device,
        dtype=next(model.parameters()).dtype
    )


    waiting = deque()


    for i, p in enumerate(prompt_strs):
        ids = ID_mapper.encode(p)
        if not ids:
            raise ValueError(f"Prompt {i} encodes to zero tokens")
        waiting.append(_Request(i, ids, max_new_tokens))


    free_lanes = list(range(num_lanes))

    prefilling: dict[int, _Request] = {}     # lane -> prompt not fully consumed yet
    running:    dict[int, _Request] = {}     # lane -> generating
    finished:   dict[int, _Request] = {}     # input index -> done


    def sample(logits):                      
        # [B,V] -> [B,1]; 
        return sample_next_token_from_logits(logits, temperature, top_k, top_p)


    def accept(lane, req, tok):
        req.generated.append(tok)
        
        if eos_token_id is not None and tok == eos_token_id:
            req.finish_reason = "eos"
        
        elif len(req.generated) >= req.max_new_tokens:
            req.finish_reason = "length"
        
        if req.finish_reason:                # retire: free the lane immediately
            cache.free(lane)
            free_lanes.append(lane)
            del running[lane]
            finished[req.index] = req


    while waiting or prefilling or running:

        # 1) ADMIT: claim a lane, no compute yet
        while waiting and free_lanes:
            prefilling[free_lanes.pop()] = waiting.popleft()

        # 2) PREFILL: one chunk of one request per iteration, so decode never stalls long
        if prefilling:
            lane, req = next(iter(prefilling.items()))
            lo = req.n_prefilled
            hi = min(lo + chunk_size, len(req.prompt_ids))
            ids    = torch.tensor([req.prompt_ids[lo:hi]], device=device)
            logits = model(ids, cache=cache, lane_ids=torch.tensor([lane], device=device))
            req.n_prefilled = hi
            if hi == len(req.prompt_ids):    # prompt done -> sample first token
                del prefilling[lane]
                running[lane] = req
                accept(lane, req, sample(logits[:, -1, :]).item())


        # 3) DECODE: one batched step over every generating request
        if running:
            lane     = list(running)
            lane_ids = torch.tensor(lane, device=device)
            last     = torch.tensor([[running[s].generated[-1]] for s in lane], device=device)
            logits   = model(last, cache=cache, lane_ids=lane_ids)
            for s, t in zip(lane, sample(logits[:, -1, :])[:, 0].tolist()):
                accept(s, running[s], t)


    results = []
    for i in range(len(prompt_strs)):
        r   = finished[i]
        gen = r.generated[:-1] if r.finish_reason == "eos" else r.generated
        results.append(Completion(
            text=ID_mapper.decode(r.prompt_ids + gen),
            completion=ID_mapper.decode(gen),
            finish_reason=r.finish_reason,
        ))
    return results
