loss curves of different model levels, and model configs, val perplexity

table of ablations (RoPE vs. none, GQA ratios, tied vs. untied embeddings, with seeds and error bars)






what I'd do next:

    Image processing Transformer that completes MNIST puzzles:

        each sample is a canvas with two MNIST digits 
        
        * if only the top layer of pixels are white     -> right digit =  (left digit + 1) mod 10
        * if the top & bottom layer of pixels are white -> right digit =  (left digit + 2) mod 10

        then add "block corruptions" to the image & train of fixing them w/ MSE loss,
        using bi-directional MHA & 2D RoPE





what I got wrong initially -> how I fiexd them:

    chunked prefill(where the cache already holds tokens & q_len > 1) mask generation was tricky 
    -> 


    
    maintaining absolute RoPE positions for sliding KV cache was tricky too. 
    -> 