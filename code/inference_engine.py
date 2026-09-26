import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange, pack

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
    Applies temperature scaling, top-k filtering, and top-p (nucleus) sampling.
    Input Shape:  [B, Vocab_Size]
    Output Shape: [B, 1]
    """
    if temperature == 0.0:
        # Greedy decoding
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 1. Apply Temperature
    logits = logits / temperature

    # 2. Top-K Filtering
    if top_k is not None and top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))


    # 3. Top-P (Nucleus) Filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right to keep the first token above top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter removed indices back to original tensor positions
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))


        # 4. Sample from Categorical Distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)            # [B, 1]
        return next_token


# ----------------------------------------------------------------------


#TODO: simplify by using sample_next_token_from_logits
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




# ----------------------------------------------------------------------

# preallocated cache 

# Fixed absolute position:
#     max total sequence length = max_context_window
#     KV cache can simply grow until that limit
@torch.inference_mode()
def advanced_inference_context_window_limited(
    model_scaffold: torch.nn.Module,
    configs: Dict[str, Any],
    num_get_steps: int,
    prompt_strs: list[str],
    device: Union[str, torch.device],
    temperature: float = 0.6,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:

    """
    used for models that have fixed positional encoding and cannot extend their generated seq
    beyond the max_context_window

    Autoregressive generation split into two distinct stages:
    1. Prefill Stage: Process all prompt tokens at once, populating the KV cache.
    2. Decode Stage: Process tokens step-by-step (1 token input per step) using cached KV pairs.
    
    prompt_tokens Shape: [B, Prompt_Len]
    Output Shape:        [B, Prompt_Len + Generated_Len]

    sliding-window KV context 
    """


    tensorized_prompts = [
        torch.tensor(
            ID_mapper.encode(prompt),
            dtype=torch.long,
        )
        for prompt in prompt_strs
    ]

    prompts = torch.stack(tensorized_prompts) # [batch_size, seq_len]


    # loaded_model = load_model_weights(model_scaffold, configs["save_path"], device)

    # loaded_model.eval() # affects batch_norm or dropout


    # a tensor for progressively appending our generated predictions to.
    # full_seq_tensor = prompt + generated tokens
    # full_seq_tensor = prompt



    # prefill

    # decode

    decoded_list = []
    for seq in prompts:
        token_ids = seq.tolist()
        decoded_list.append(
            ID_mapper.decode(token_ids)
        )
    return decoded_list


    

# ----------------------------------------------------------------------


# Ring-buffer

@dataclass
class KV_cache():
    ...


# RoPE + sliding window:
#     total generated length can exceed max_context_window
#     KV cache keeps only the newest W tokens
#     position IDs continue increasing
# cache_mode=full -> keeps the KV cache for total seq
@torch.inference_mode()
def advanced_inference(
    model_scaffold: torch.nn.Module,
    configs: Dict[str, Any],
    num_get_steps: int,
    prompt_strs: list[str],
    device: Union[str, torch.device],
    cache_mode: str = "sliding_window",
    temperature: float = 0.6,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:

    ...

    # ID_map prompts & stack

    # load model

    # manage KV: 

    # sample 

    # de-ID the seq


    #---
    
    # prefill()

    # for gen_step in num_get_steps-1:
    #    decode()