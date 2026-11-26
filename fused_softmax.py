import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

def naive_softmax(x):
    # input shape: [M, N]
    x_max = x.max(dim=1)[0] # shape: [M] read  MN elements ; write M  elements
    # dim=1：按行求最大值，对每一行，找出该行最大的值
    # x.max(dim=1) 返回 (values, indices)，[0] 取最大值，丢弃下标
    z = x - x_max[:, None]  # shape: [M, N] read MN + M elements ; write MN elements
    numerator = torch.exp(z)  # shape: [M, N] read  MN elements ; write MN elements
    denominator = numerator.sum(dim=1)  # shape: [M] read  MN elements ; write M  elements
    out = numerator / denominator[:, None]  # shape: [M, N] read MN + M elements ; write MN elements
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return out

@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    ):
    PID = tl.program_id(0)
    row_start = PID
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start,n_rows,row_step,num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0,BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=float('-inf'))

        row_minus_max = row - tl.max(row,axis=0)
        numerator = tl.exp(row_minus_max)
        denomenator = tl.sum(numerator,axis=0)
        softmax_output = numerator / denomenator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr+col_offsets, softmax_output, mask=mask)

properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"] # 128
NUM_REGS = properties["max_num_regs"] # 65536
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] # 101376
WARP_SIZE = properties["warpSize"] # 32
# print(f"NUM_SM: {NUM_SM}")
# print(f"NUM_REGS per SM: {NUM_REGS}")
# print(f"TOTAL_SRAM_PER_SM (bytes): {TOTAL_SRAM_PER_SM}, KB: {TOTAL_SRAM_PER_SM/1024:.1f}")
# print(f"WARP_SIZE: {WARP_SIZE}")

def softmax(x):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM >= 200000 else 2

    y = torch.empty_like(x)
    kernel = _softmax_kernel.warmup(
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=(1,)
    )
    kernel._init_handles()
    n_regs_pre_program = kernel.n_regs
    sram_needed_pre_program = kernel.metadata.shared

    reg_occupancy = NUM_REGS // (n_regs_pre_program * WARP_SIZE * num_warps)
    # NUM_REGS = 65536
    # each program might use:
        # n_regs_per_program = 32
        # WARP_SIZE = 32
        # num_warps = 8
    # so each program needs (n_regs_per_program * WARP_SIZE * num_warps) registers total
    # 65536 // (32 * 32 * 8) = 8 programs per SM
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_pre_program
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs = min(NUM_SM * programs_per_sm, n_rows)
    grid = (num_programs,1,1)
    kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols
    )
    # x: [M, N] -> x.stride(): [N, 1]
    # z.shape: [B,N,D] -> z.stride(): [N*D, N, 1]
    return y

def test_softmax_kernel(size:tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    assert isinstance(size, tuple) and len(size) == 2
    torch.manual_seed(0)
    x = torch.randn(size[0], size[1], device=device)
    z_tri = softmax(x)
    z_ref = torch.softmax(x,axis=1)
    z_naive = naive_softmax(x)
    torch.testing.assert_close(z_naive, z_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 50)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == "__main__":
    test_softmax_kernel(size=(1823, 781))
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path="results",print_data=True, show_plots=False)