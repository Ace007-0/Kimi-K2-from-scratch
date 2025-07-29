# Kimi-K2-from-scratch

This project is a simplified and educational implementation of Kimi K2 built entirely from scratch, inspired by Karpathy’s nanoGPT. It integrates modern techniques such as Mixture of Experts (MoE), SwiGLU activation, MuonClip optimizer, and custom tokenization, trained on the TinyStories dataset.

## Project Goals

1. Recreate Kimi-K2 architecture without using pre-trained models or tokenizers
   
2. Implement core components like:-
    1). Transformer blocks
   
    2). Mixture of Experts (MoE) routing
   
    3). SwiGLU activation
   
    4). MuonClip optimizer (custom AdamW + gradient clipping)
   
3. Train on a small dataset (TinyStories)

## Core Architecture

1. Tokenizer
A simple word-level tokenizer built with regex:
      re.findall(r"\w+|[^\w\s]", text.lower())
This converts the dataset into a vocabulary of lowercase words and punctuation.

2. Model Components
   
   1). Embedding
   token_embed → turns word indices into dense vectors
   pos_embed → adds positional information

   2).Transformer Block
   Uses Multi-Head Attention with causal masking to prevent peeking into future tokens.
   Residual connections and layer normalization included.

   3). MoE Layer (Mixture of Experts)
   Multiple parallel expert MLPs
   A router dynamically selects top-k experts per token.
   Each token gets routed only to the top-scoring experts.

    4). SwiGLU Activation
   More expressive than ReLU
   Defined as: SwiGLU(x) = x₁ * SiLU(x₂) (split on channel dimension)

   5). Output
   A linear layer projects back to vocabulary size for token prediction

3. MuonClip Optimizer

      A custom optimizer based on AdamW with gradient clipping:

      1). Clipping gradients when norm exceeds threshold
 
      2). Adds weight decay
 
      3). Fully implemented manually to show inner workings

## Training

1). Trained on 100k TinyStories samples

2). Batch size: 32 | Block size: 256 | Steps: 2000

3). Loss is plotted every 100 steps

## Results
generate(model, "Once upon a time", max_new_tokens=100)
Generates plausible story-like continuations. As training steps increase, story coherence improves.

## Installation & Run

pip install datasets torch matplotlib tqdm --quiet
Just paste and run the code in Google Colab, and training begins automatically.

## Dataset
We use roneneldan/TinyStories, a collection of child-friendly stories designed to train small language models.

## Why This Project Matters

1). This is a great educational replica of Kimi-K2 that:

2). Requires no deep learning frameworks like Hugging Face Transformers

3). Helps you understand every layer from tokenization to logits

4). Includes cutting-edge components like MoE, SwiGLU, and custom optimizers

## Future Work

1). Replace word-level tokenizer with BPE or SentencePiece (optional)

2). Add multi-GPU training and evaluation scripts

3). Train longer on larger subsets for better quality

Acknowledgments

Karpathy’s nanoGPT

Vizuara.ai — for inspiration and educational content in AI and ML

MoonshotAI’s Kimi-K2

Ronen Eldan’s TinyStories dataset
