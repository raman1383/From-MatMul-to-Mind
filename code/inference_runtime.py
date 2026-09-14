# get model weights + prompt + [prefill, decode] -> KV cache + output seq

import os
import torch


def inference_runtime(configs, device, max_decode_steps:int):

    # load model
    save_path = configs["save_path"]
    if os.path.exists(save_path):
        state_dict = torch.load(save_path, map_location=device)
        fresh_model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {save_path}")
        print("Model Parameters Loaded:")
       
        for name, param in fresh_model.named_parameters():
            print(f" -> {name}: shape {list(param.shape)}")
    
    else:
        print(f"Error: Weight file '{save_path}' not found!")
        return


    model.eval()
    with torch.no_grad():
        ...

    # prefill
    # kv_cache = prefill(prompt, model)

    # decode
    # for i in range(max_decode_steps):
    #     decode(kv_cache)



