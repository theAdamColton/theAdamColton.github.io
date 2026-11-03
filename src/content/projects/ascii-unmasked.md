---
title: Ascii Art Latent Masked Transformer Model
url: https://github.com/theAdamColton/ascii-unmasked
date: 2022-12-31
cover: ascii-unmasked-demo2.mp4
coverCaption: >-
  Interpolation of the latent space. The top left corner shows the discrete
  tokens. The left ascii art is the decoded representation of the current
  embedding. On the right is the original.
---

I trained a variational quantized autoencoder on ASCII art. The characters in
the art are represented as one hot encoded vectors at each 'pixel' in the
string. The discrete latents learned by this model were used to train a
bidirectional transformer. The transformer was trained to predict masked latent
tokens, akin to [MaskGit](https://arxiv.org/pdf/2202.04200.pdf) by Google
research.
