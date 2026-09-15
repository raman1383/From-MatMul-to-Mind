# get model weights + prompt + [prefill, decode] -> KV cache + output seq

import os
import torch
from einops import rearrange
from tokenizerModule import Tokenizer
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# @torch.inference_mode()
# def inference(
#         model_scaffold:torch.nn.Module, 
#         configs:Dict[str, Any], 
#         device:Union[str, torch.device], 
#         prompt:str, 
#         max_new_tokens:int, 
#         temperature: float = 1.0,
#         top_k: Optional[int] = 50,
#         top_p: Optional[float] = None,
#         eos_token_id: Optional[int] = 83,
#     )->str:


#     if max_new_tokens < 0:
#         raise ValueError("num_gen_steps must be non-negative")


#     # load model
#     model = load_model_weights(model_scaffold, configs["save_path"], device)
#     if model is None:
#         raise RuntimeError(f"Failed to load weights from {configs['save_path']}")
#     model.eval()


#     prompt_ids: List[int] = Tokenizer.encode(prompt)
#     if not prompt_ids:
#         raise ValueError("Prompt encoded to an empty sequence")


#     # TODO: truncate to not exceede max seq length
#     input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T)


#     #     # Prefill: Tokenize and initialize sequence context
#     #     kv_cache = prefill(prompt_as_IDs, configs, device)

#     #     # Decode: Generate new tokens step-by-step
#     #     decode(prompt_as_IDs, kv_cache, model, configs, device, temperature, top_k)

#     # # Decode tokens back into human-readable text
#     # # generated_tokens = context_state["generated_tokens"]
#     # return Tokenizer.decode(generated_tokens)



def load_model_weights(
    model_scaffold: torch.nn.Module,
    save_path: str,
    device: torch.device,
    ) -> Optional[torch.nn.Module]:

    if not os.path.isfile(save_path):
        print(f"[ERROR] Weight file not found: {save_path}")
        return None

    try:
        state_dict = torch.load(save_path, map_location=device)
        model_scaffold.load_state_dict(state_dict)
        model_scaffold = model_scaffold.to(device)
        model_scaffold.eval()

        print(f"[INFO] Successfully loaded weights from {save_path}")
        # Optional: print parameter summary only once
        n_params = sum(p.numel() for p in model_scaffold.parameters())
        print(f"[INFO] Model has {n_params:,} parameters")
        return model_scaffold
    
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        return None


def prefill(prompt: str, configs: dict, device: str) -> dict:
    ...

def decode(KV_cache: dict, model, configs: dict, device: str, temperature: float = 1.0, top_k: int = 10):
    ...

def sample():
    ...

@dataclass
class KVCache:
    """
    Explicit container for key/value tensors across layers.

    Design goals
    ------------
    - Clear ownership and lifetime
    - Easy to inspect / debug
    - Works for models with 0, 1, or many attention layers
    - Supports both classic multi-head and GQA layouts
    """

    # List of (key, value) pairs, one entry per layer that has attention.
    # For a pure embedding model this list is empty.
    layers: List[Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    # Current sequence length that the cache represents
    seq_len: int = 0

    def is_empty(self) -> bool:
        return len(self.layers) == 0




@torch.inference_mode()
def simple_and_naive_inference(model_scaffold:torch.nn.Module, 
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

    prompt_to_IDs = Tokenizer.encode(prompt)
    if len(prompt_to_IDs) == 0:
        raise ValueError("[ERR] Prompt produced no tokens. Provide a non-empty prompt.")

    loaded_model = load_model_weights(model_scaffold, configs["save_path"], device)


    # a list for progressively appending our generated predictions to
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
            print(context_window_limited_seq_tensor.shape)
            print(f"[INFO] Step {step+1}/{num_gen_steps}: Context window length: {context_window_limited_seq_tensor.shape[1]}")

            # (batch_size=1, seq_len=len(context_window_limited_seq), vocab_size)
            model_output_logits = loaded_model(context_window_limited_seq_tensor)
            print(model_output_logits.shape)

            # extract the logits for ONLY the final token position in the sequence
            last_token_logits = rearrange(model_output_logits[:, -1, :], "1 vocab_size -> vocab_size")
            print(last_token_logits.shape)

            # temperature controls the sharpness of the distribution
            # tempered_logits = last_token_logits / max(temperature, 1e-6)

            # convert to probability distribution
            probabilities = torch.softmax(last_token_logits, dim=-1)
            print(probabilities.shape)
            print(probabilities)

            # multinomial sampling
            next_token_tensor = torch.multinomial(probabilities, num_samples=1)
            print("next_token_tensor.shape= ")
            print(next_token_tensor.shape)
            print(next_token_tensor)
            next_token = next_token_tensor.item()

            # Append predicted token to sequence for the next autoregressive loop step
            full_seq.append(next_token)
            print("full_seq.append after append: ")
            print(full_seq)

    return Tokenizer.decode(full_seq)




