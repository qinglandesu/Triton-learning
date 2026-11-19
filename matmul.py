import torch
import triton
import triton.language as tl
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

"""
# Do in parallel
for m in range(0, M, BLOCK_SIZE_M):
    # Do in parallel
    for n in range(0, N, BLOCK_SIZE_N):
        acc = torch.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
            b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
            acc += dot(a, b)
        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
"""

autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
]

@triton.autotune(
    configs=autotune_configs,
    key=['M','N','K'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_stride_M, a_stride_K,
    b_stride_K, b_stride_N,
    c_stride_M, c_stride_N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ):
    PID = tl.program_id(0)
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    group_id = PID // num_PID_in_group
    first_PID_in_group_along_M = group_id * GROUP_SIZE
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE)
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj

    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)

    a_offsets = offsets_M[:,None] * a_stride_M + offsets_K[None,:] * a_stride_K
    b_offsets = offsets_K[:,None] * b_stride_K + offsets_N[None,:] * b_stride_N
    # [:,None] turns [m1,m2,m3] into [[m1],[m2],[m3]]
    # [None,:] turns [n1,n2,n3] into [[m1],[m2],[m3]]
    # [[m1n1,m1n2,m1n3],
    #  [m2n1,m2n2,m2n3],
    #  [m3n1,m3n2,m3n3]]
    accumulator = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)

    for k in range(0,tl.cdiv(K,BLOCK_SIZE_K)):
        mask = offsets_K < K - k * BLOCK_SIZE_K # offsets_K + k*BLOCK_SIZE_K < K
        a = tl.load(a_ptr + a_offsets, mask=mask[None,:], other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=mask[:,None], other=0.0)

        accumulator = tl.dot(a, b, acc=accumulator)

        a_offsets += BLOCK_SIZE_K * a_stride_K
        b_offsets += BLOCK_SIZE_K * b_stride_K

    accumulator = accumulator.to(tl.float16)

    c_offsets = offsets_M[:,None] * c_stride_M + offsets_N[None,:] * c_stride_N
    c_mask = (offsets_M[:,None] < M) & (offsets_N[None,:] < N)
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)

def matmul(a,b):
    assert a.ndim == b.ndim == 2
    assert a.shape[1] == b.shape[0]
    (M, K), (_, N)  = a.shape, b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    return c


def test_matmul_kernel(size:tuple,atol=1e-2,rtol=1e-1,device=DEVICE):
    torch.manual_seed(0)
    assert isinstance(size, tuple) and len(size) == 2
    a = torch.randn(size,device=DEVICE,dtype=torch.float16)
    b = torch.randn(size,device=DEVICE,dtype=torch.float16)

    c_tri = matmul(a,b)
    c_ref = torch.matmul(a,b)

    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print("PASSED")

configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"], # we can increase multiple dimensions simultaneously while benchmarking
        x_vals = [128 * i for i in range(2, 33)],
        line_arg = "provider", 
        line_vals = ["torch", "triton"],
        line_names = ["PyTorch", "Triton"],
        styles = [("green", "-"), ("blue", "-")],
        ylabel = "TFLOPS", 
        plot_name = "matmul-performance",
        args={},
    )
]
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 3 * M * N * K * 1e-12 / (ms * 1e-3)
        # 3 = number of memory operations (2 read + 1 write)
        # M * N * K = number of elements per memory op
        # 1e-12 converts flops to Teraflops
        # 1e-3 converts milliseconds to seconds
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    test_matmul_kernel(size=(512,512))

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path="results",print_data=True, show_plots=False)