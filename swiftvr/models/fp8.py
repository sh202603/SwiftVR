"""FP8 (E4M3) linears for the DiT blocks (opt-in, ``prepare_for_inference(fp8=True)``).

Weights are quantized once with a per-tensor scale; activations get a dynamic
per-tensor scale on every call, computed on the GPU (no host sync) by two Triton
kernels (abs-max reduction, then scale + saturate + cast). The GEMM itself is
``torch._scaled_mm`` (cuBLASLt), which on GeForce Blackwell runs ~3.5x faster
than the bf16 GEMM (bf16 with fp32 accumulation is half rate there) and halves
the weight memory.

Measured on SwiftVR, per-channel weight scales and MXFP8 32-element block
scales gave the same error as per-tensor scales (the e4m3 mantissa dominates),
so the simplest scheme is used.

The quantization kernels follow flashvsr-sm89-ops
(https://github.com/aireet/flashvsr-sm89-ops, Apache-2.0).
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
_BLOCK = 8192
_GELU_C0 = 0.7978845608028654  # sqrt(2/pi), as in aten's tanh GELU


@triton.jit
def _gelu_tanh(x, C0: tl.constexpr):
    # tanh(u) = 1 - 2 / (exp(2u) + 1); rounded to bf16 like aten's GELU output
    u = C0 * (x + 0.044715 * x * x * x)
    g = 0.5 * x * (2.0 - 2.0 / (tl.exp(2.0 * u) + 1.0))
    return g.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _load_act(x_ptr, b_ptr, offs, mask, N, HAS_BIAS: tl.constexpr, GELU: tl.constexpr, C0: tl.constexpr):
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        # bf16 add, rounded like eager's `x + bias`
        x = (x + tl.load(b_ptr + offs % N, mask=mask, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    if GELU:
        x = _gelu_tanh(x, C0)
    return tl.where(mask, x, 0.0)


@triton.jit
def _absmax_kernel(x_ptr, b_ptr, amax_ptr, n, N, HAS_BIAS: tl.constexpr, GELU: tl.constexpr,
                   C0: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = _load_act(x_ptr, b_ptr, offs, offs < n, N, HAS_BIAS, GELU, C0)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(x), axis=0))


@triton.jit
def _quant_kernel(x_ptr, b_ptr, amax_ptr, scale_ptr, out_ptr, n, N, HAS_BIAS: tl.constexpr,
                  GELU: tl.constexpr, C0: tl.constexpr, FP8_MAX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    s = tl.maximum(tl.load(amax_ptr), 1e-12) / FP8_MAX
    if pid == 0:
        tl.store(scale_ptr, s)
    x = _load_act(x_ptr, b_ptr, offs, mask, N, HAS_BIAS, GELU, C0)
    # saturate: the float -> e4m3fn cast turns overflow into NaN
    v = tl.minimum(tl.maximum(x / s, -FP8_MAX), FP8_MAX)
    tl.store(out_ptr + offs, v.to(tl.float8e4nv), mask=mask)


def quantize_fp8(x: torch.Tensor, bias: torch.Tensor = None, gelu: bool = False):
    """``x`` contiguous bf16 ``[M, K]`` -> (FP8 ``[M, K]``, fp32 scale on the GPU).
    Quantizes ``gelu_tanh(x + bias)`` (each part optional) without materializing it."""
    n = x.numel()
    amax = torch.zeros(1, dtype=torch.float32, device=x.device)
    scale = torch.empty((), dtype=torch.float32, device=x.device)
    out = torch.empty(x.shape, dtype=FP8, device=x.device)
    grid = (triton.cdiv(n, _BLOCK),)
    has_bias = bias is not None
    b = bias if has_bias else x
    N = x.shape[-1]
    _absmax_kernel[grid](x, b, amax, n, N, HAS_BIAS=has_bias, GELU=gelu, C0=_GELU_C0,
                         BLOCK=_BLOCK, num_warps=8)
    _quant_kernel[grid](x, b, amax, scale, out, n, N, HAS_BIAS=has_bias, GELU=gelu, C0=_GELU_C0,
                        FP8_MAX=FP8_MAX, BLOCK=_BLOCK, num_warps=8)
    return out, scale


class FP8Linear(nn.Module):
    """Drop-in for a bf16 ``nn.Linear``: per-tensor FP8 weight, dynamic per-tensor
    FP8 activation, bf16 output.

    The bias is not given to ``_scaled_mm``: with a bias epilogue cuBLASLt picks
    an sm89 kernel that is ~1.7x slower on sm120 than the bias-free one."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach()
        if w.shape[0] % 16 or w.shape[1] % 16:
            raise ValueError(f"FP8 GEMM needs dims divisible by 16, got {tuple(w.shape)}")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        w_scale = (w.abs().amax().float() / FP8_MAX).clamp(min=1e-12)
        # Stored as integer views: Module.to(dtype=...) casts every floating
        # buffer (float8 included) and would destroy the FP8 codes / fp32 scale.
        self.register_buffer("qweight", (w.float() / w_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8).view(torch.uint8))
        self.register_buffer("w_scale", w_scale.view(torch.int32))
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().to(torch.bfloat16))

    def mm(self, x8, x_scale):
        """GEMM without the bias."""
        return torch._scaled_mm(x8, self.qweight.view(FP8).t(), scale_a=x_scale,
                                scale_b=self.w_scale.view(torch.float32), out_dtype=torch.bfloat16)

    def forward(self, x):
        shape = x.shape
        y = self.mm(*quantize_fp8(x.reshape(-1, shape[-1]).contiguous()))
        if self.bias is not None:
            y.add_(self.bias)
        return y.view(*shape[:-1], self.out_features)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class FP8FeedForward(nn.Module):
    """Replacement for diffusers ``FeedForward(activation_fn="gelu-approximate")``:
    FP8 ``proj -> GELU(tanh) -> out`` with the GELU fused into the quantization of
    the second GEMM's input, so the bf16 GELU output is never materialized."""

    def __init__(self, ff):
        super().__init__()
        act, _drop, out = ff.net
        if getattr(act, "approximate", None) != "tanh":
            raise ValueError("FP8FeedForward expects a tanh-approximate GELU FeedForward")
        self.proj = FP8Linear(act.proj)
        self.out = FP8Linear(out)

    def forward(self, x):
        shape = x.shape
        h = self.proj.mm(*quantize_fp8(x.reshape(-1, shape[-1]).contiguous()))
        y = self.out.mm(*quantize_fp8(h, bias=self.proj.bias, gelu=True))
        if self.out.bias is not None:
            y.add_(self.out.bias)
        return y.view(*shape[:-1], self.out.out_features)


def fp8_supported(device) -> bool:
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 9)


def convert_blocks_to_fp8(model) -> int:
    """Swap every transformer block's attention projections and FFN for FP8
    versions, one module at a time (the bf16 original is freed as soon as its
    FP8 copy exists). Must run after ``fuse_projections``. Returns the number of
    GEMMs converted."""
    n = 0
    for blk in model.blocks:
        blk = getattr(blk, "_orig_mod", blk)
        for attn in (blk.attn1, blk.attn2):
            for name in ("to_qkv", "to_q", "to_kv"):
                lin = getattr(attn, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(attn, name, FP8Linear(lin))
                    n += 1
            attn.to_out[0] = FP8Linear(attn.to_out[0])
            n += 1
        blk.ffn = FP8FeedForward(blk.ffn)
        n += 2
    return n
