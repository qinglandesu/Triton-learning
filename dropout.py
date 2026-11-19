import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def dropout_kernel(
    x_ptr, x_keep_ptr, output_ptr,
    n_elements, p,
    BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1-p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta:(triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    dropout_kernel[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output

x = torch.randn(size = (10,), device=DEVICE)
p = 0.5
x_keep = (torch.rand(size = (10,), device=DEVICE) > p).to(torch.int32)
output = dropout(x, x_keep, p)
print("dropout")
print(x)
print(x_keep)
print(output)

@triton.jit
def seeded_dropout_kernel(
    x_ptr, output_ptr,
    n_elements, p, seed,
    BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1-p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta:(triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    seeded_dropout_kernel[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output

x = torch.randn(size=(10, ), device=DEVICE)
# Compare this to the baseline - dropout mask is never instantiated!
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)
print("seeded_dropout")
print(x)
print(output)
print(output2)
print(output3)