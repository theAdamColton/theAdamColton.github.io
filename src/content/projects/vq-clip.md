---
title: VQ-CLIP
url: https://github.com/theAdamColton/vq-clip
date: 2023-05-19
---

So, you've heard about [CLIP](https://arxiv.org/abs/2103.00020), but have you
heard about CLIP but with a <span class="rainbow-text-animated">quantized
embedding space</span>??

Rather than using a real number vector as the embedding, VQ-CLIP uses a list of
k integer class IDs. The embedding codes from these models can be used for
exciting downstream tasks, such as autoregressive CLIP embedding generation.
