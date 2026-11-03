---
title: spiht-py
url: https://github.com/theAdamColton/spiht-py
date: 2023-12-21
cover: spiht/motorcycle.gif
coverCaption: >-
  Left: intermediate decoded image. Right: intermediate coefficients from the
  discrete wavelet transform. Bits per pixel (BPP) are shown in the top left
  corner. As BPP increases, you can see how the encoder assigns information to
  coefficients at higher frequencies.
---

An implementation of the [SPIHT](https://spiht.com/) algorithm in Rust, with
Python bindings. SPIHT is a lossy image compression algorithm. Like JPG, it can
reduce the amount of bits required to store images. Unlike JPG, the SPIHT
bitstream can be interrupted at any point and the entire image decoded. There
are no 'blocks' in the SPIHT algorithm.
