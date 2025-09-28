import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=1)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_online_rowwise_kernel(
    x_ptr,
    y_ptr,
    stride_x,
    stride_y,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_x_ptr = x_ptr + row_id * stride_x
    row_y_ptr = y_ptr + row_id * stride_y

    cols = tl.arange(0, BLOCK_SIZE)
    m = tl.full([1], -float('inf'), dtype=tl.float32)
    s = tl.zeros([1], dtype=tl.float32)

    col_start = 0
    while col_start < n_cols:
        offs = col_start + cols
        mask = offs < n_cols
        x = tl.load(row_x_ptr + offs, mask=mask, other=-float('inf'))
        x_f32 = tl.cast(x, tl.float32)
        m_chunk = tl.max(x_f32, axis=0)
        m_new = tl.maximum(m, m_chunk)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x_f32 - m_new), axis=0)
        m = m_new
        col_start += BLOCK_SIZE

    inv_s = 1.0 / s
    col_start = 0
    while col_start < n_cols:
        offs = col_start + cols
        mask = offs < n_cols
        x = tl.load(row_x_ptr + offs, mask=mask, other=-float('inf'))
        x_f32 = tl.cast(x, tl.float32)
        y = tl.exp(x_f32 - m) * inv_s
        tl.store(row_y_ptr + offs, y, mask=mask)
        col_start += BLOCK_SIZE


def triton_softmax(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    assert x.ndim == 2, "Input must be 2D"
    assert dim == 1, "This implementation supports softmax over dim=1 (rows)"
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert x.dtype == torch.float32, "This implementation expects float32 inputs"
    x = x.contiguous()
    n_rows, n_cols = x.shape
    y = torch.empty_like(x, dtype=torch.float32)

    BLOCK_SIZE = 4096
    num_warps = 8 if BLOCK_SIZE >= 2048 else 4

    softmax_online_rowwise_kernel[(n_rows,)](
        x, y,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs a Softmax activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x, dim=1)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32).to("cuda")
    return [x]

def get_init_inputs():
    return []

def test_latency(model1, model2, inputs, warmup=10, iters=100):
    import time
    model1.eval()
    model2.eval()
    # warm-up
    for _ in range(warmup):
        out1 = model1(*inputs)
        out2 = model2(*inputs)
    # sync CUDA before timing
    torch.cuda.synchronize()
    start1 = time.time()
    for _ in range(iters):
        _ = model1(*inputs)
    torch.cuda.synchronize()
    end1 = time.time()
    avg_time1 = (end1 - start1) * 1000 / iters  # ms
    torch.cuda.synchronize()
    start2 = time.time()
    for _ in range(iters):
        _ = model2(*inputs)
    torch.cuda.synchronize()
    end2 = time.time()
    avg_time2 = (end2 - start2) * 1000 / iters  # ms
    print(f"Model1 (PyTorch) avg latency: {avg_time1:.4f} ms")
    print(f"Model2 (Triton)  avg latency: {avg_time2:.4f} ms")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    inputs = get_inputs()
    model1 = Model().cuda()
    model2 = ModelNew().cuda()
    test_latency(model1, model2, inputs)
