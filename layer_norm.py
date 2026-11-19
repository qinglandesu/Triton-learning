import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _layernorm_forward_1(
    x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr,
    stride_M, N, eps,
    BLOCK_SIZE: tl.constexpr,
    ):
    row = tl.program_id(0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M

    sum_acc = tl.zeros([BLOCK_SIZE],dtype=tl.float32)
    for offset in range(0,N,BLOCK_SIZE):
        cols = offset+tl.arange(0,BLOCK_SIZE)
        mask = cols < N
        x_ptrs = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_acc += x_ptrs
    
    mean = tl.sum(sum_acc,axis=0) / N

    diff_acc = tl.zeros([BLOCK_SIZE],dtype=tl.float32)
    for offset in range(0,N,BLOCK_SIZE):
        cols = offset+tl.arange(0,BLOCK_SIZE)
        mask = cols < N
        x_ptrs = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x_ptrs - mean, 0.0)
        diff_acc += diff * diff
    
    var = tl.sum(diff_acc,axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    for offset in range(0,N,BLOCK_SIZE):
        cols = offset + tl.arange(0,BLOCK_SIZE)
        mask = cols < N
        w_ptrs = tl.load(w_ptr + cols, mask=mask)
        b_ptrs = tl.load(b_ptr + cols, mask=mask)
        x_ptrs = tl.load(x_ptr + cols, mask=mask)

        x_normed = (x_ptrs - mean) * rstd
        y = x_normed * w_ptrs + b_ptrs

        tl.store(y_ptr + cols, y, mask=mask)

@triton.jit
def _layernorm_forward_2(
    x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr,
    stride_M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M

    sum_val = 0.0
    sum_sq_val = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq_val += tl.sum(x * x, axis=0)
    mean = sum_val / N
    var = sum_sq_val / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(y_ptr + cols, y, mask=mask)

@triton.jit
def _layernorm_backward_dLdx(
    x_ptr, dLdx_ptr, dLdy_ptr,
    w_ptr, dLdw_inter_ptr, dLdb_inter_ptr,
    mean_ptr, rstd_ptr,
    locks_ptr,
    stride, N,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
    ):
    PID = tl.program_id(0)
    cols = tl.arange(0,BLOCK_SIZE_N)
    mask = cols < N
    x_ptr += PID * stride
    dLdx_ptr += PID * stride
    dLdy_ptr += PID * stride

    x = tl.load(x_ptr+cols, mask=mask, other=0.0).to(tl.float32)
    dLdy = tl.load(dLdy_ptr+cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr+cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + PID)
    rstd = tl.load(rstd_ptr + PID)

    x_normed = tl.where(mask, (x-mean)*rstd, 0.0)
    dydx_normed = tl.where(mask, w*dLdy, 0.0)
    c1 = tl.sum(x_normed * dydx_normed, axis=0) / N
    c2 = tl.sum(dydx_normed, axis=0) / N
    dLdx = (dydx_normed - (x_normed * c1 + c2)) * rstd

    tl.store(dLdx_ptr+cols, dLdx, mask=mask)

    dLdw_contri = (dLdy * x_normed).to(w.dtype)
    dLdb_contri = (dLdy).to(w.dtype)

    lock_id = PID % GROUP_SIZE
    locks_ptr += lock_id
    count_ptr = locks_ptr + GROUP_SIZE

    dLdw_inter_ptrs = dLdw_inter_ptr + lock_id * N + cols
    dLdb_inter_ptrs = dLdb_inter_ptr + lock_id * N + cols

    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass
        # - If it's 0 (unlocked), change it to 1 (locked) and return 0 (False) to exit the while loop
        # - If it's 1 (already locked), leave it as 1 and return 1 (True) so that we stay in the while loop
    
    count = tl.load(count_ptr)
    if count == 0: # if this PID is the first one to access the lock
        tl.atomic_xchg(count_ptr, 1)
    else:
        dLdw_contri += tl.load(dLdw_inter_ptrs, mask=mask) # we load and add in one step (+= operator)
        dLdb_contri += tl.load(dLdb_inter_ptrs, mask=mask)

    tl.store(dLdw_inter_ptrs, dLdw_contri, mask=mask)
    tl.store(dLdb_inter_ptrs, dLdb_contri, mask=mask)
    
    tl.debug_barrier() # 保证当前 program 的所有线程完成写入后再解锁
    tl.atomic_xchg(locks_ptr, 0) # unlock

@triton.jit
def _layernorm_backward_dLdw_dLdb(
    dLdw_inter_ptr, dLdb_inter_ptr,
    dLdw_ptr, dLdb_ptr,
    GROUP_SIZE, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
    ):
    PID = tl.program_id(0)
    col_offsets = PID * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)

    dLdw_acc = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N), dtype=tl.float32)
    dLdb_acc = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, GROUP_SIZE, BLOCK_SIZE_M):
        row_offsets = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_offsets[:,None] < GROUP_SIZE) & (col_offsets[None,:] < N)
        offsets = row_offsets[:,None] * N + col_offsets[None,:]

        dLdw_acc += tl.load(dLdw_inter_ptr + offsets, mask=mask, other=0.0)
        dLdb_acc += tl.load(dLdb_inter_ptr + offsets, mask=mask, other=0.0)
    
    dLdw_chunk = tl.sum(dLdw_acc, axis=0)
    dLdb_chunk = tl.sum(dLdb_acc, axis=0)

    tl.store(dLdw_ptr + col_offsets, dLdw_chunk, mask=col_offsets < N)
    tl.store(dLdb_ptr + col_offsets, dLdb_chunk, mask=col_offsets < N)

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x, normalized_shape,
        weight, bias, eps
        ):
        M,N = x.reshape(-1,x.shape[-1]).shape
        mean = torch.empty((M,),dtype=torch.float32,device=x.device)
        rstd = torch.empty((M,),dtype=torch.float32,device=x.device) # reciprocal std
        y = torch.empty_like(x)

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE,triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        _layernorm_forward_2[(M,)](
            x, y,
            weight, bias,
            mean, rstd,
            x.stride(0), N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x,weight,bias,mean,rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps

        return y
    
    @staticmethod
    def backward(
        ctx,
        dLdy
        ):
        x,w,b,mean,rstd = ctx.saved_tensors
        M,N = x.reshape(-1,x.shape[-1]).shape

        dLdx = torch.empty_like(x) # (M,N)
        dLdw = torch.empty_like(w) # (N,)
        dLdb = torch.empty_like(b) # (N,)

        GROUP_SIZE = 64
        if N <= 8192: GROUP_SIZE = 96
        if N <= 4096: GROUP_SIZE = 128
        if N <= 1024: GROUP_SIZE = 256

        # 这里如果设置成 x.dtype，即 FP16，中间结果每次写回都会被截成 FP16，精度会不满足atol
        dLdw_inter = torch.zeros((GROUP_SIZE,N),dtype=torch.float32,device=w.device)
        dLdb_inter = torch.zeros((GROUP_SIZE,N),dtype=torch.float32,device=b.device)

        locks = torch.zeros(2 * GROUP_SIZE,dtype=torch.int32,device=w.device)

        _layernorm_backward_dLdx[(M,)](
            x, dLdx, dLdy,
            w, dLdw_inter, dLdb_inter,
            mean, rstd,
            locks,
            x.stride(0), N,
            GROUP_SIZE=GROUP_SIZE,
            BLOCK_SIZE_N = ctx.BLOCK_SIZE,
            num_warps = ctx.num_warps
        )

        grid = lambda meta:[triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _layernorm_backward_dLdw_dLdb[grid](
            dLdw_inter, dLdb_inter, dLdw, dLdb,
            min(GROUP_SIZE,M), N,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128
        )

        return dLdx, None, dLdw, dLdb, None

layernorm = LayerNorm.apply

def test_layernorm_kernel(M,N, dtype, eps=1e-5, device=DEVICE):
    x = -2.3 + 0.5 * torch.randn((M,N),dtype=dtype,device=device)
    x.requires_grad_(True)
    weight = torch.rand((N,),dtype=dtype,device=device,requires_grad=True)
    bias = torch.rand((N,),dtype=dtype,device=device,requires_grad=True)
    y_tri = layernorm(x,(N,),weight,bias,eps)
    y_ref = torch.nn.functional.layer_norm(x,(N,),weight,bias,eps).to(dtype)
    torch.testing.assert_close(y_tri,y_ref,atol=1e-2,rtol=0)
    print("PASSED forward")

    dLdy = 0.1 * torch.randn_like(x)

    y_tri.backward(dLdy,retain_graph=True)
    dLdx_tri,dLdw_tri,dLdb_tri = [_.grad.clone() for _ in [x,weight,bias]]
    x.grad,weight.grad,bias.grad = None,None,None

    y_ref.backward(dLdy,retain_graph=True)
    dLdx_ref,dLdw_ref,dLdb_ref = [_.grad.clone() for _ in [x,weight,bias]]

    torch.testing.assert_close(dLdx_tri, dLdx_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_ref, atol=1e-2, rtol=0)
    # rtol=0 means we don't use relative tolerance 
    print("PASSED backward")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-forward',  # 图名字不一样
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'forward'},
    ))
def benchmark_fwd(M, N, dtype, provider, mode='forward', eps=1e-5, device=DEVICE):
    x_shape = (M, N)
    w_shape = (N, )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    x.requires_grad_(True)
    quantiles = [0.5, 0.05, 0.95]

    def y_fwd():
        if provider == "triton":
            return layernorm(x, w_shape, weight, bias, eps)
        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)

    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(torch.randn_like(x), retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)], # if you increase past 32 the kernel will break since features become larger than 64kb
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'}, # so we're actually only benchmarking the backward pass
    ))
def benchmark_bwd(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (N, )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)#, requires_grad=True)
    dLdy = .1 * torch.randn_like(x)
    x.requires_grad_(True) 
        # setting this here instead of x's initial definition means the graph doesn't have to move through the -2.3 and 0.5 operations
    quantiles = [0.5, 0.05, 0.95]

    def y_fwd():
        if provider == "triton":
            return layernorm(x, w_shape, weight, bias, eps) 
        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps) 

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dLdy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__=="__main__":
    test_layernorm_kernel(1151, 18192, torch.float16) # N 不能大于32768

    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--benchmarkfwd":
            benchmark_fwd.run(save_path="results",print_data=True, show_plots=False)
        elif sys.argv[1] == "--benchmarkbwd":
            benchmark_bwd.run(save_path="results",print_data=True, show_plots=False)