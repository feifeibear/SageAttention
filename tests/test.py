import torch
import torch.nn.functional as F
from sageattention.core import sageattn
import time
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_kvpacked_func
import os
from flash_attn.flash_attn_interface import _flash_attn_forward

def test_sage_vs_flash():
    # Set up test parameters
    batch_size = 1
    num_heads = 32
    seq_len = 1024
    head_dim = 64
    
    # current_cuda_device = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(current_cuda_device)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    dtype = torch.float16
    # Generate random input tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)

    test_attention(q, k, v, is_causal=False)

def test_attention(q, k, v, is_causal):

    o, lse = sageattn(q, k, v, is_causal=is_causal, ret_lse=True)

    q_flash = q.transpose(1, 2)  # [B, S, H, D]
    k_flash = k.transpose(1, 2)
    v_flash = v.transpose(1, 2)
    block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
        q_flash,
        k_flash,
        v_flash,
        dropout_p = 0,
        softmax_scale = q_flash.shape[-1] ** (-0.5),
        causal=is_causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )   
    block_out = block_out.transpose(1, 2)
    print("LSE DIFF:")
    print(lse - block_lse)

    print("OUT DIFF:")
    print(f"Max absolute difference: {torch.max(torch.abs(o - block_out))}")
    print(f"Mean absolute difference: {torch.mean(torch.abs(o - block_out))}")

def standard_attention(q, k, v, is_causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

if __name__ == "__main__":
    test_sage_vs_flash()
