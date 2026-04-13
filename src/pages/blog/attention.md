---
layout: ../../layouts/Layout.astro
title: "Breaking down Scaled Dot-Product Attention"
date: 2026-04-14
---

# Breaking down Scaled Dot-Product Attention
<span class="date">2026-04-14</span>

When working with Transformers, the core mechanism is scaled dot-product attention. The formula is elegantly simple:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Here is a quick PyTorch implementation to build intuition:

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn
