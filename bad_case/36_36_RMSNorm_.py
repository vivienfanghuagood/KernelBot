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

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_reduce_normalize_kernel(
    x_ptr,            # *T
    out_ptr,          # *T
    N,                # int
    C,                # int (num_features, reduction dim)
    S,                # int (spatial size = prod of remaining dims after C)
    stride_n,         # int (stride for dim 0)
    stride_c,         # int (stride for dim 1, reduction dim)
    eps,              # float32
    BLOCK_C: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N * S:
        return

    # Map program id to (n, s)
    n = pid // S
    s = pid % S

    base = n * stride_n + s

    # First pass: compute sum of squares across C
    acc = 0.0
    c0 = 0
    while c0 < C:
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        ptrs = x_ptr + base + offs_c * stride_c
        x = tl.load(ptrs, mask=mask_c, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
        c0 += BLOCK_C

    inv_rms = 1.0 / tl.sqrt(acc / C + eps)

    # Second pass: normalize and store
    c0 = 0
    while c0 < C:
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        ptrs = x_ptr + base + offs_c * stride_c
        x = tl.load(ptrs, mask=mask_c, other=0.0)  # keep original dtype for cast on store
        y = (x.to(tl.float32) * inv_rms).to(OUT_DTYPE)
        tl.store(out_ptr + base + offs_c * stride_c, y, mask=mask_c)
        c0 += BLOCK_C


def triton_rmsnorm_channel(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    RMSNorm along channel dimension (dim=1) for 4D tensors [N, C, H, W].
    Falls back to PyTorch when not supported.
    """
    if x.dim() != 4:
        # Fallback to torch for non-4D inputs
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
        return x / rms

    # Ensure CUDA and contiguous
    if not x.is_cuda:
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
        return x / rms

    x_contig = x.contiguous()
    N, C, H, W = x_contig.shape
    S = H * W

    out = torch.empty_like(x_contig)

    stride_n = x_contig.stride(0)
    stride_c = x_contig.stride(1)

    # Choose block size over channel dim (power of two up to 256)
    if C >= 256:
        BLOCK_C = 256
    elif C >= 128:
        BLOCK_C = 128
    elif C >= 64:
        BLOCK_C = 64
    elif C >= 32:
        BLOCK_C = 32
    else:
        BLOCK_C = 16

    # Map torch dtype to triton dtype for store
    if x_contig.dtype == torch.float16:
        out_dtype = tl.float16
        num_warps = 4
    elif x_contig.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
        num_warps = 4
    elif x_contig.dtype == torch.float32:
        out_dtype = tl.float32
        num_warps = 4
    else:
        # Unsupported dtype -> fallback
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
        return x / rms

    grid = (N * S,)
    rmsnorm_reduce_normalize_kernel[grid](
        x_contig,
        out,
        N,
        C,
        S,
        stride_n,
        stride_c,
        float(eps),
        BLOCK_C=BLOCK_C,
        OUT_DTYPE=out_dtype,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized RMSNorm using a fused Triton kernel over the channel dimension.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel on CUDA 4D tensors; otherwise fallback to PyTorch
        if x.is_cuda and x.dim() == 4:
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
