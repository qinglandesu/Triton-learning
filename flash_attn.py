import torch
import triton
import triton.language as tl
import math

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
        DIAGONAL: tl.constexpr, # blocks on the DIAGONAL get special treatment if this is set to true
        offsets_QO_N: tl.constexpr, offsets_KV_N: tl.constexpr,
        N: tl.constexpr, Dh: tl.constexpr
    ):
    if DIAGONAL: # Used only for the blocks along the diagonal in which there is transition between non-masked and masked keys
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * BLOCK_SIZE_QO
        
        lo = tl.multiple_of(lo, BLOCK_SIZE_QO)
        # tl.multiple_of(x, align)：告诉 Triton 编译器，x一定是这个对齐数的倍数
    else: # blocks below the diagonal
        lo, hi = 0, block_index_QO * BLOCK_SIZE_QO

    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo

    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)
        
        # compute (Q @ K^T) / sqrt(Dh)
        mask_KV_N = offsets_KV_N < N
        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.0) # shape (Dh, BLOCK_SIZE_KV)
        S = tl.dot(Q, K_T) * scale # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
        
        if DIAGONAL:
            causal_mask = offsets_QO_N[:, None] >= (offsets_KV_N[None, :])
            S += tl.where(causal_mask, 0, -1.0e6) # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV) # S = tl.where(causal_mask, S, -1.0e6)
        
        M_new = tl.maximum(M, tl.max(S, axis=1)) # shape (BLOCK_SIZE_QO)
        S -= M_new[:, None] # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
        
        P = tl.exp2(S) # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
        L_new = tl.sum(P, axis=1) # shape (BLOCK_SIZE_QO)
        alpha = tl.exp2(M - M_new) # shape (BLOCK_SIZE_Q)
        L = L * alpha + L_new # shape (BLOCK_SIZE_QO)
        
        # compute O = P @ V + O * alpha
        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.) # shape (BLOCK_SIZE_KV, Dh)
        O = O * alpha[:, None] # shape (BLOCK_SIZE_QO, Dh)
        O = tl.dot(P, V, acc=O) # shape (BLOCK_SIZE_QO, Dh)
        
        M = M_new

        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV

    return O, L, M

@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config(
            {"BLOCK_SIZE_QO": BLOCK_SIZE_QO, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages, num_warps=num_warps,
        )
        for BLOCK_SIZE_QO in [16]#, 32, 64, 128]
        for BLOCK_SIZE_KV in [16]#, 32, 64, 128]
        for num_stages in [3]#, 5, 7]
        for num_warps in [4]#, 8, 16]
    ],
    key=["Dh"],
)
@triton.jit
def attn_fwd(
    Q_ptr, K_ptr,  V_ptr,   # each shape (B, H, N, Dh)
    O_ptr,                  # shape (B, H, N, Dh)
    LSE_ptr,                # shape (B, H, N). here we first store the max values of each row & later the logsumexp trick
    scale,                  # 1/sqrt(Dh)
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
    stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
    stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    B, # unlike other tensor dimensions, batch size can be more flexible for runtime differences
    H, N, # H: tl.constexpr, N: tl.constexpr, 
    Dh: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    ):
    rln2: tl.constexpr = 1.4426950408889634 # 1/math.log(2)
    scale *= rln2 # 用来替代exp: e^x = 2^(x * log2(e))

    index_BH = tl.program_id(axis=1) # 第1维：batch+head 打包
    index_B = index_BH // H
    index_H = index_BH % H
    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H

    block_index_QO = tl.program_id(axis=0) # 第0维：query块编号
    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0,BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0,BLOCK_SIZE_KV)
    offsets_Dh = tl.arange(0, Dh)

    Q_offsets = offsets_QO_N[:,None] * stride_Q_N + offsets_Dh[None,:] * stride_Q_Dh # shape (BLOCK_SIZE_QO, Dh)
    K_T_offsets = offsets_Dh[:, None] * stride_K_Dh + offsets_KV_N[None, :] * stride_K_N # shape (Dh, BLOCK_SIZE_KV) 加载的时候直接做转置
    V_offsets = offsets_KV_N[:, None] * stride_V_N + offsets_Dh[None, :] * stride_V_Dh # shape (BLOCK_SIZE_KV, Dh)
    
    mask_QO_N = offsets_QO_N < N
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:,None], other=0.0) # shape (BLOCK_SIZE_QO, Dh)
    # 当前最大值 the running maximum
    M = tl.full(shape=[BLOCK_SIZE_QO], value=-1e6, dtype=tl.float32) # large negative number will get ignored by tl.max()
    # 当前归一化分母 the running sum
    L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32) # 1 is because we'll be using exponentials and e^0=1
    # 累加结果 the accumulator for the output, which is a group of rows of the O matrix
    O = tl.zeros([BLOCK_SIZE_QO, Dh], dtype=tl.float32)

    # 第一遍：对角线以下的KV块（完全可见，无mask）
    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        False, # blocks on the DIAGONAL get special treatment if this is set to true
        offsets_QO_N, offsets_KV_N,
        N, Dh
    )
    # 第二遍：对角线KV块（需要上三角mask，causal）
    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        True, # blocks on the DIAGONAL get special treatment if this is set to true
        offsets_QO_N, offsets_KV_N,
        N, Dh
    )

    O = O / L[:, None] # shapes (BLOCK_SIZE_QO, Dh) / (BLOCK_SIZE_QO, 1) = (BLOCK_SIZE_QO, Dh)
    LSE = M + tl.math.log2(L) # shape (BLOCK_SIZE_QO)
    
    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask) # shape (BLOCK_SIZE_QO)

    O_offsets = (offsets_QO_N[:, None] * stride_O_N + offsets_Dh[None, :] * stride_O_Dh)
    tl.store(O_ptr + O_offsets, O, mask=mask_QO_N[:,None]) # shape (BLOCK_SIZE_Q, Dh)

class _flashattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,scale):
        assert q.shape == k.shape == v.shape
        assert q.shape[-1] <= 128, \
            f'flash attention only supports head dimension of 128 less but got {q.shape[-1]}'
            # the kernel actually isn't this limited but too much larger and i think it might overwhelm SRAM
        B, H, N, Dh = q.shape
        assert q.device == k.device and q.device == v.device
        assert q.dtype == k.dtype == v.dtype == torch.float32

        B,H,N,Dh = q.shape
        O = torch.empty_like(q)
        LSE = torch.empty((B,H,N), device=DEVICE, dtype=torch.float32)

        grid = lambda args:(
            triton.cdiv(N,args["BLOCK_SIZE_QO"]),
            B * H
        )

        attn_fwd[grid](
            q,k,v,O,LSE,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, N, Dh,
        )

        ctx.save_for_backward(q,k,v,O,LSE)
        ctx.grid = grid
        ctx.B,ctx.H,ctx.N,ctx.Dh = B,H,N,Dh
        ctx.scale = scale
        return O

triton_attention = _flashattention.apply

def test_flashattention_kernel(B,H,N,Dh, device=DEVICE, atol=5e-3):
    q = torch.randn((B,H,N,Dh), dtype=torch.float32, device=DEVICE, requires_grad=True)
    k = torch.randn((B,H,N,Dh), dtype=torch.float32, device=DEVICE, requires_grad=True)
    v = torch.randn((B,H,N,Dh), dtype=torch.float32, device=DEVICE, requires_grad=True)
    scale = 1 / math.sqrt(Dh)
    tri_out = triton_attention(q,k,v,scale)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
    torch.testing.assert_close(tri_out,ref_out,atol=atol,rtol=0)
    print("PASSED forward")

# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]: # , "bwd"
    configs.append(
        triton.testing.Benchmark(
            x_names=["SEQ_LEN"],
            x_vals=[512 * i for i in range(1, 9)], # LOWER IF YOU DON'T HAVE ENOUGH RAM
            line_arg="provider",
            line_vals=["torch", "triton"],
            line_names=["Torch", "Triton"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name=f"attention-performance-{mode}",
            args={"mode": mode},
        ))

@triton.testing.perf_report(configs)
def bench_flash_attention(SEQ_LEN, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float32
    BATCH, N_HEADS = 32, 4 # LOWER THESE IF YOU DON'T HAVE ENOUGH RAM
    HEAD_DIM = 128 # AND THIS IF YOU DON"T HAVE ENOUGH SRAM
    q = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    sm_scale = 1 / math.sqrt(HEAD_DIM)
    if provider == 'torch':
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    if provider == 'triton':
        fn = lambda: triton_attention(q, k, v, sm_scale)
    if mode == "bwd":
        O = fn()
        dLdO = torch.randn_like(O)
        fn = lambda: O.backward(dLdO, retain_graph=True)
    ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = 2 * flops_per_matmul * 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)

if __name__ == "__main__":
    # always run unit-tests
    test_flashattention_kernel(1, 1, 128, 32) # without block masking
    test_flashattention_kernel(1, 1, 128, 64) # without block masking
    test_flashattention_kernel(1, 1, 128, 128) # without block masking
    test_flashattention_kernel(32, 8, 69, 128) # with block masking

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        bench_flash_attention.run(save_path="results",print_data=True, show_plots=False)