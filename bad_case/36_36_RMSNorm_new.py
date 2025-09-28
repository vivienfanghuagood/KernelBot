
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Calculate the RMS along the feature dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

        # Normalize the input by dividing by the RMS
        return x / rms

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 32,  "SINGLE_PASS": True},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_C": 64,  "SINGLE_PASS": True},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 128, "SINGLE_PASS": True},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_C": 256, "SINGLE_PASS": True},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_C": 64,  "SINGLE_PASS": False}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 128, "SINGLE_PASS": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_C": 256, "SINGLE_PASS": False}, num_warps=8, num_stages=3),
    ],
    key=["C"],
)
@triton.jit
def rmsnorm_kernel(
    x_ptr,            # *T
    out_ptr,          # *T
    N,                # int
    C,                # int (num_features, reduction dim)
    H,                # int
    W,                # int
    stride_n,         # int (stride for dim 0)
    stride_c,         # int (stride for dim 1, reduction dim)
    stride_h,         # int (stride for dim 2)
    stride_w,         # int (stride for dim 3)
    eps,              # float32
    BLOCK_C: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    pid = tl.program_id(0)
    HW = H * W
    if pid >= N * HW:
        return

    # Map program id to (n, h, w)
    n = pid // HW
    hw = pid % HW
    h = hw // W
    w = hw % W

    base = n * stride_n + h * stride_h + w * stride_w

    offs_c = tl.arange(0, BLOCK_C)
    tl.max_contiguous(offs_c, BLOCK_C)

    # If SINGLE_PASS and C fits in BLOCK_C, do a single read/compute/write pass
    if SINGLE_PASS and (C <= BLOCK_C):
        mask = offs_c < C
        ptrs = x_ptr + base + offs_c * stride_c
        x = tl.load(ptrs, mask=mask, other=0.0)
        x_f = x.to(tl.float32)
        ss = tl.sum(x_f * x_f, axis=0)
        inv_rms = 1.0 / tl.sqrt(ss / C + eps)
        y = (x_f * inv_rms).to(OUT_DTYPE)
        tl.store(out_ptr + base + offs_c * stride_c, y, mask=mask)
        return

    # General path: two-pass when C > BLOCK_C or SINGLE_PASS is False
    acc = 0.0
    c0 = 0
    while c0 < C:
        offs = c0 + offs_c
        mask = offs < C
        x = tl.load(x_ptr + base + offs * stride_c, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
        c0 += BLOCK_C

    inv_rms = 1.0 / tl.sqrt(acc / C + eps)

    c0 = 0
    while c0 < C:
        offs = c0 + offs_c
        mask = offs < C
        x = tl.load(x_ptr + base + offs * stride_c, mask=mask, other=0.0)
        y = (x.to(tl.float32) * inv_rms).to(OUT_DTYPE)
        tl.store(out_ptr + base + offs * stride_c, y, mask=mask)
        c0 += BLOCK_C


def triton_rmsnorm_channel(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    RMSNorm along channel dimension (dim=1) for 4D tensors [N, C, H, W].
    Falls back to PyTorch when not supported.
    """
    if x.dim() != 4 or not x.is_cuda:
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
        return x / rms

    # Ensure CUDA and at least has a well-defined layout
    x_in = x
    if not (x_in.is_contiguous() or x_in.is_contiguous(memory_format=torch.channels_last)):
        x_in = x_in.contiguous()

    N, C, H, W = x_in.shape
    out = torch.empty_like(x_in)

    stride_n = x_in.stride(0)
    stride_c = x_in.stride(1)
    stride_h = x_in.stride(2)
    stride_w = x_in.stride(3)

    # Map torch dtype to triton dtype for store and choose num_warps via autotune
    if x_in.dtype == torch.float16:
        out_dtype = tl.float16
    elif x_in.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif x_in.dtype == torch.float32:
        out_dtype = tl.float32
    else:
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
        return x / rms

    grid = (N * H * W,)

    rmsnorm_kernel[grid](
        x_in,
        out,
        N,
        C,
        H,
        W,
        stride_n,
        stride_c,
        stride_h,
        stride_w,
        float(eps),
        OUT_DTYPE=out_dtype,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized RMSNorm using a fused, autotuned Triton kernel over the channel dimension.
    Supports both NCHW contiguous and channels_last contiguous tensors on CUDA.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dim() == 4 and x.size(1) == self.num_features:
            return triton_rmsnorm_channel(x, self.eps)
        else:
            rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
            return x / rms


# Replicate original helpers for completeness
batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).to("cuda")
    return [x]

def get_init_inputs():
    return [features]

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
    model1 = Model(*get_init_inputs()).cuda()
    model2 = ModelNew(*get_init_inputs()).cuda()
    test_latency(model1, model2, inputs)
