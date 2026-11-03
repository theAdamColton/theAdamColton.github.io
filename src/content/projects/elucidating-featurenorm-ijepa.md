---
title: Elucidating the Role of Feature Normalization in IJEPA
url: https://arxiv.org/abs/2508.02829
date: 2025-07-24
cover:
  - elucidating-the-role-of-fn-in-ijepa/loss-maps.avif
  - elucidating-the-role-of-fn-in-ijepa/feature-maps.avif
links:
  paper: https://arxiv.org/abs/2508.02829
  code: https://github.com/theAdamColton/elucidating-featurenorm-ijepa
---

In the standard image joint embedding predictive architecture (IJEPA),
features at the output of the teacher encoder are layer normalized (LN) before
serving as a distillation target for the student encoder and predictor. We
propose that this feature normalization disrupts the natural energy hierarchy
of visual tokens, where high-energy tokens (those with larger L2 norms) encode
semantically important image regions. LN forces all features to have identical
L2 norms, effectively equalizing their energies and preventing the model from
prioritizing semantically rich regions. We find that IJEPA models trained with
feature LN exhibit loss maps with significant checkerboard-like artifacts. We
propose that feature LN be replaced with a DynTanh activation as the latter
better preserves token energies and allows high-energy tokens to greater
contribute to the prediction loss. We show that IJEPA trained with feature
DynTanh exhibits a longer-tailed loss distribution and fixes the checkerboard
artifacts in the loss map. Our empirical results show that our simple
modification improves ImageNet linear probe accuracy from 38% to 42.7% for
ViT-Small and reduces RMSE by 0.08 on NYU Depth V2 monocular depth estimation.
These results suggest that preserving natural token energies is crucial for
effective self-supervised visual representation learning.
