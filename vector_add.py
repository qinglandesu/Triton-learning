import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ):
    PID = tl.program_id(axis = 0)
    block_start = PID * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask = mask, other = None)
    y = tl.load(y_ptr + offsets, mask = mask, other = None)
    out = x + y

    tl.store(out_ptr + offsets, out, mask = mask)



def add(x, y):
    assert x.device == DEVICE and y.device == DEVICE

    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # cdiv(m, n) = (m + n - 1) // n (ceiling division with rounding)

    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)

    return out

def test_add_kernel(size, atol=1e-3, rtol=1e-3,device=DEVICE):
    torch.manual_seed(0)

    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)
    z_tri = add(x,y)
    z_ref = x + y

    torch.testing.assert_close(z_tri,z_ref,atol=atol,rtol=rtol)
    print(z_tri)
    print(z_ref)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(z_tri - z_ref))}')
    print("passed")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12,24)],
        x_log = True,
        line_arg='provider',
        line_vals=['triton','torch'],
        line_names=['Triton','Torch'],
        styles=[('blue','-'),('red','-')],
        ylabel='GB/s',
        plot_name='vector_add_performance',
        args={},
    )
)
def benchmark(size,provider):
    x=torch.randn(size, device=DEVICE,dtype=torch.float32)
    y=torch.randn(size, device=DEVICE,dtype=torch.float32)

    quantiles = [0.5,0.05,0.95]

    if provider == 'torch':
        ms, min_mis, max_mis = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_mis, max_mis = triton.testing.do_bench(lambda: add(x,y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms),gbps(min_mis),gbps(max_mis)

if __name__ == '__main__':
    test_add_kernel(2000)
    test_add_kernel(4096)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path="results",print_data=True, show_plots=False)