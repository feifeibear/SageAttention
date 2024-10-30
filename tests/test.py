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
    num_heads = 8
    seq_len = 3816
    head_dim = 64
    
    # current_cuda_device = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(current_cuda_device)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    dtype = torch.float16
    # Generate random input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)

    test_attention(q, k, v, is_causal=False)

def test_attention(q, k, v, is_causal):

    q_sage = q.transpose(1, 2)  # [B, S, H, D]
    k_sage = k.transpose(1, 2)
    v_sage = v.transpose(1, 2)
    q_scale = 1 / q.shape[-1] ** (-0.5)
    o, lse = sageattn(q_sage, k_sage, v_sage, is_causal=is_causal, ret_lse=True)
    o = o.transpose(1, 2)
    # lse = lse.squeeze(dim=-1).transpose(1, 2)

    block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        dropout_p = 0,
        softmax_scale = q.shape[-1] ** (-0.5),
        causal=is_causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )   
    print("LSE DIFF:")
    print(f"block_lse {block_lse.shape}")

    if lse is not None:
        print(f"lse {lse.shape}")
        print("lse:")
        print(lse)
        print("block_lse:")
        print(block_lse)

    print("OUT DIFF:")
    print(f"Max absolute difference: {torch.max(torch.abs(o - block_out))}")
    print(f"Mean absolute difference: {torch.mean(torch.abs(o - block_out))}")

def standard_attention(q, k, v, is_causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

if __name__ == "__main__":
    test_sage_vs_flash()
