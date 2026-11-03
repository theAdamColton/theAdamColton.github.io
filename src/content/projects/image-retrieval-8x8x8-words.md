---
title: "Image Retrieval: A Picture is worth 8x8x8 Words"
url: /a-picture-is-worth-8x8x8-words
date: 2023-04-11
cover:
  - 2d_kgrams.png
  - image-ret-poster.avif
coverCaption: >-
  2D kgrams can be used to produce global representations which capture spatial
  relationships
---

SOTA image retrieval models usually use global real-number embeddings, obtained
from big neural networks. But why is there no love for more traditional
information retrieval techniques such as BoW? Given a 256x256 image, we encode
it into a 8x8x8 matrix of discrete integer tokens. We do this using a ResNet
model trained with a learned
[vector quantization](https://github.com/lucidrains/vector-quantize-pytorch)
layer. Using these tokens, we use <span class="rainbow-text-animated">2D
Kgrams</span> to obtain global vectors containing term frequencies.
