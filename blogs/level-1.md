Level 1: Tokenization & Embedding

first, lets take a look at our dataset:

the text dataset is a part of the "Tiny Stories" dataset which is a bunch of small & simple stories, 

these are what's called "special tokens", this dataset was made by another LLM chatbot & these tokens are used to tell the LLM were a document ends & a new one begins, & the LLM uses these tokens to say it has finished generating the answer to a prompt.

here is our training objective:
* given a sequence as input, output that sequence shifted one token to the right
![alt text](../media/autoregressive_gen_steps.png)

---

Tokenization:

Vectors are the universal language, during Tokenization we map sub-words, image & audio chunks into vectors, so our model can process them. 


But here is an engeneering trade-off:

why not map an entire sentence/image into just one token ID? 
why not map a  single letter/pixel   into just one token ID? 

It is a balancing act between average length and content capacity: 

If the chunk of text or image is too big(too many letter or pixels) the embed_dim sized vector cannot represent its content well enough, so the model cannot learn the hidden relationships of the data...

If the chunk is too small(a single letter or pixel) then tokenizing an image or sentence results in a large number of tokens, and this lengthy token sequence is very expensive for our transformer to process, because it needs to let each token communicate w/ every other token in the same sequence(cost: O(n^2))...

---

Tokenizing Text:


Now let's write the code to tokenize text:

Ideally we want the most commonly occurring letter groups to be packaged-up into one token, that way the average length of any sequence in our dataset will be shorter:

for example:
hello, he, hell, hesitate, -> 'he'

Now let's write an algorithm that:

1. counts the frequencies of all adjacent pairs of symbols
2. merges the most frequently co-occurring pairs 

the repeat this process until we hit the limit of our vocabulary size.

vocabulary size is the number of unique symbols our model can emit, a larger number makes the model more expressive(entropy rate = log 1/vocab) but also more expensive(the output liner layer is 'embed_dim' by 'vocab_size')


---

Bigram Language Model:


