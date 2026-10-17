"""Streaming wrapper around the Restoration-aware Autoencoder.

It runs the encoder/decoder clip-by-clip while passing the MemBlock and TPool
boundary state across chunks, so the result is identical to encoding/decoding
the whole clip at once.

With ``fused=True`` (CUDA only) activations stay channels-last (NHWC) end to end,
so cuDNN needs no NCHW<->NHWC transposes around each conv, and conv+bias+ReLU /
conv+bias+residual+ReLU run as single cuDNN calls with fused epilogues. The
model itself is unchanged; only the execution differs (fp32 epilogues instead of
bf16-rounded intermediates, i.e. rounding-level differences).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.reae import MemBlock, TPool, TGrow
from .chunk import ChunkSpec, ChunkType

_CL = torch.channels_last


def _is_cl(x):
    return x.dim() == 4 and not x.is_contiguous() and x.is_contiguous(memory_format=_CL)


def _conv_relu(conv, x):
    """relu(conv(x) + bias) as one cuDNN call."""
    return torch.cudnn_convolution_relu(x, conv.weight, conv.bias, conv.stride, conv.padding,
                                        conv.dilation, conv.groups)


def _memblock_fused(b, x, past):
    """MemBlock.forward with the three convs' bias/ReLU and the residual add in
    cuDNN epilogues."""
    c0, c2, c4 = b.conv[0], b.conv[2], b.conv[4]
    h = _conv_relu(c0, torch.cat([x, past], 1))
    h = _conv_relu(c2, h)
    return torch.cudnn_convolution_add_relu(h, c4.weight, b.skip(x), 1.0, c4.bias, c4.stride,
                                            c4.padding, c4.dilation, c4.groups)


def _tgrow2_fused(tg, x):
    """TGrow with stride 2 as a 1x1 conv. The stock path repeats each frame twice
    (nearest) and applies a (3,1,1) conv with zero padding, so output frame j of
    input frame x is (W1+W2)x for j=0 and (W0+W1)x for j=1."""
    w = tg.conv3d.weight[:, :, :, 0, 0].float()                      # [C, C, 3]
    w = torch.cat([w[..., 1] + w[..., 2], w[..., 0] + w[..., 1]], 0)  # [2C, C]
    y = F.conv2d(x, w.to(x.dtype)[:, :, None, None].contiguous(memory_format=_CL))
    NT, SC, H, W = y.shape
    C = SC // 2
    # [NT, 2C, H, W] (channels j-major) -> [2NT, C, H, W], keeping NHWC memory
    y = y.permute(0, 2, 3, 1).reshape(NT, H, W, 2, C).permute(0, 3, 1, 2, 4).reshape(NT * 2, H, W, C)
    return y.permute(0, 3, 1, 2)


def _is_up2_conv3(up, conv):
    """nearest x2 Upsample followed by a stride-1 3x3 conv with zero padding 1."""
    return (isinstance(up, nn.Upsample) and up.mode == "nearest" and up.size is None
            and up.scale_factor in (2, 2.0) and isinstance(conv, nn.Conv2d)
            and conv.kernel_size == (3, 3) and conv.stride == (1, 1) and conv.padding == (1, 1)
            and conv.dilation == (1, 1) and conv.groups == 1 and conv.padding_mode == "zeros")


def _fold_up2_weight(w):
    """[Co, Ci, 3, 3] -> [4Co, Ci, 2, 2], one 2x2 kernel per output phase (a, b).
    After nearest x2 upsampling, output row 2i+a of the 3x3 conv reads input
    rows (i-1, i) with taps (W0, W1+W2) for a=0 and rows (i, i+1) with taps
    (W0+W1, W2) for a=1; columns likewise."""
    def taps(w, a, dim):
        w0, w1, w2 = w.unbind(dim)
        return torch.stack([w0, w1 + w2] if a == 0 else [w0 + w1, w2], dim)
    w = w.float()
    return torch.cat([taps(taps(w, a, 2), b, 3) for a in (0, 1) for b in (0, 1)], 0)


def _up2_conv3_fused(conv, x, relu):
    """conv(Upsample(x, 2, nearest)) [+ ReLU] at the input resolution: one 2x2
    conv (padding 1) for all four output phases, then a strided copy of each
    phase into the channels-last output. 16 instead of 36 MACs per input pixel
    and channel pair, and the 4x upsampled input is never materialized."""
    key = (conv.weight.data_ptr(), conv.weight._version, x.dtype, x.device)
    cache = getattr(conv, "_up2_fold", None)
    if cache is None or cache[0] != key:
        k = _fold_up2_weight(conv.weight).to(x.dtype).contiguous(memory_format=_CL)
        bias = None if conv.bias is None else conv.bias.to(x.dtype).repeat(4)
        cache = conv._up2_fold = (key, k, bias)
    _, k, bias = cache
    if relu:
        y = torch.cudnn_convolution_relu(x, k, bias, (1, 1), (1, 1), (1, 1), 1)
    else:
        y = F.conv2d(x, k, bias, padding=1)
    # y[:, phase (a, b), p, q] is output pixel (2(p-a)+a, 2(q-b)+b)
    N, _, H, W = x.shape
    Co = conv.out_channels
    out = torch.empty((N, Co, 2 * H, 2 * W), dtype=y.dtype, device=y.device, memory_format=_CL)
    for a in (0, 1):
        for b in (0, 1):
            c = (2 * a + b) * Co
            out[:, :, a::2, b::2].copy_(y[:, c:c + Co, a:a + H, b:b + W])
    return out


def run_frame_batches(fn, x, frame_batch_size=None):
    """Apply a frame-independent ``fn`` to ``x`` (``[F, C, H, W]``) a few frames
    at a time, writing into one preallocated output so that only one batch's
    intermediate activations are alive at once. ``fn`` may emit a fixed number
    of output frames per input frame (e.g. TGrow)."""
    n = x.shape[0]
    if not frame_batch_size or n <= frame_batch_size:
        return fn(x)
    out = None
    for s in range(0, n, frame_batch_size):
        e = min(s + frame_batch_size, n)
        y = fn(x[s:e])
        if out is None:
            k = y.shape[0] // (e - s)
            out = torch.empty((n * k, *y.shape[1:]), dtype=y.dtype, device=y.device,
                              memory_format=_CL if _is_cl(y) else torch.contiguous_format)
        out[s * k:e * k].copy_(y)
        del y
    return out


def _run_layers(layers, fused=False):
    layers = list(layers)

    def fn(x):
        use_fused = fused and x.is_cuda
        i = 0
        while i < len(layers):
            layer = layers[i]
            if use_fused and i + 1 < len(layers) and _is_up2_conv3(layer, layers[i + 1]):
                relu = i + 2 < len(layers) and isinstance(layers[i + 2], nn.ReLU)
                x = _up2_conv3_fused(layers[i + 1], x, relu)
                i += 3 if relu else 2
                continue
            if (use_fused and isinstance(layer, nn.Conv2d) and i + 1 < len(layers)
                    and isinstance(layers[i + 1], nn.ReLU)):
                x = _conv_relu(layer, x)
                i += 2
                continue
            if use_fused and isinstance(layer, TGrow) and layer.stride == 2:
                x = _tgrow2_fused(layer, x)
            else:
                x = layer(x)
            i += 1
        return x
    return fn


def _group_frames(xs, stride, channels_last):
    """[N, n, C, H, W] -> [N*n/stride, stride*C, H, W], channel = j*C + c for the
    j-th frame of each group (TPool's input layout)."""
    N, n, C, H, W = xs.shape
    if not channels_last:
        return xs.reshape(N * n // stride, stride * C, H, W)
    if stride == 1:
        return xs.reshape(N * n, C, H, W).contiguous(memory_format=_CL)
    g = xs.permute(0, 1, 3, 4, 2).reshape(N * n // stride, stride, H, W, C)
    g = g.permute(0, 2, 3, 1, 4).reshape(N * n // stride, H, W, stride * C)
    return g.permute(0, 3, 1, 2)


def apply_parallel_with_boundary(model, x, state=None, frame_batch_size=None, fused=False):
    """Run ``model`` (a Sequential of streaming blocks) over ``x``.

    ``x`` has shape ``[N, T, C, H, W]``. ``state`` carries the MemBlock/TPool
    boundary buffers from the previous chunk; the updated state is returned.
    Runs of stateless layers between MemBlock/TPool are executed
    ``frame_batch_size`` frames at a time (``None`` = all frames at once).
    ``fused`` selects the channels-last / fused-cuDNN execution (CUDA only).
    """
    if state is None:
        state = {}
    new_state = {}
    fused = fused and x.is_cuda
    N, T, C, H, W = x.shape
    x = x.reshape(N * T, C, H, W)
    if fused:
        x = x.contiguous(memory_format=_CL)

    layers = list(model)
    i = 0
    while i < len(layers):
        b = layers[i]
        if not isinstance(b, (MemBlock, TPool)):
            j = i
            while j < len(layers) and not isinstance(layers[j], (MemBlock, TPool)):
                j += 1
            x = run_frame_batches(_run_layers(layers[i:j], fused), x, frame_batch_size)
            i = j
            continue

        if isinstance(b, MemBlock):
            NT, C, H, W = x.shape
            T_ = NT // N
            _x = x.view(N, T_, C, H, W)
            key = f"mem_{i}"
            # past[t] = x[t-1]; built in x's memory layout (NCHW or NHWC)
            mem = torch.empty_like(x)
            mv = mem.view(N, T_, C, H, W)
            mv[:, 1:].copy_(_x[:, :-1])
            if key in state:
                mv[:, :1].copy_(state[key])
            else:
                mv[:, :1].zero_()
            new_state[key] = _x[:, -1:].detach().clone()
            x = _memblock_fused(b, x, mem) if fused else b(x, mem)

        elif isinstance(b, TPool):
            NT, C, H, W = x.shape
            T_ = NT // N
            _x = x.view(N, T_, C, H, W)
            key = f"tpool_{i}"
            if key in state and state[key] is not None:
                _x = torch.cat([state[key], _x], dim=1)
                T_ = _x.shape[1]
            n_full = (T_ // b.stride) * b.stride
            rem = T_ - n_full
            new_state[key] = _x[:, n_full:].detach().clone() if rem > 0 else None
            if n_full > 0:
                groups = _group_frames(_x[:, :n_full], b.stride, fused)
                x = run_frame_batches(b.conv, groups, frame_batch_size)
            else:
                return None, new_state
        i += 1

    NT, C, H, W = x.shape
    return x.view(N, NT // N, C, H, W), new_state


def to_channels_last(model):
    """Store every Conv2d weight channels-last (in place), so the fused path's
    convs take NHWC weights directly instead of converting them per call."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = m.weight.data.contiguous(memory_format=_CL)
    return model


class StreamingTAE:
    """Chunk-wise ReAE encode/decode with causal state carried across chunks.

    The decoder is split after its last TGrow: the head holds all cross-frame
    state, the tail is frame-independent. Frames discarded on the first decode
    are dropped before the tail, so the full-resolution layers never compute
    them. ``frame_batch_size`` bounds how many frames the stateless layers run
    at once (``None`` = whole chunk); lower values trade a little speed for
    peak memory.
    """

    def __init__(self, model, frame_batch_size=None, fused=True):
        self.model = model
        self.frame_batch_size = frame_batch_size
        self.fused = fused
        split = max(i for i, m in enumerate(model.decoder) if isinstance(m, TGrow)) + 1
        self._dec_head = model.decoder[:split]
        self._dec_tail = _run_layers(model.decoder[split:], fused)
        self._enc_st = None
        self._dec_st = None
        self._enc_left = None
        self._first_dec = True

    def reset(self):
        self._enc_st = self._dec_st = None
        self._enc_left = None
        self._first_dec = True

    # ----- Fixed-size chunk interface (offline, frame-count preserving) ----- #

    @torch.no_grad()
    def encode_chunk_fixed(self, x: torch.Tensor, spec: ChunkSpec) -> torch.Tensor:
        ps = self.model.patch_size
        if ps > 1:
            N, T, C, H, W = x.shape
            x = F.pixel_unshuffle(x.reshape(N * T, C, H, W), ps)
            x = x.reshape(N, T, *x.shape[1:])

        if spec.ctype == ChunkType.LAST:
            x = torch.cat([x, x[:, -1:].expand(-1, 3, -1, -1, -1)], dim=1)

        z, self._enc_st = apply_parallel_with_boundary(
            self.model.encoder, x, self._enc_st, self.frame_batch_size, self.fused)
        return z

    @torch.no_grad()
    def decode_chunk_fixed(self, z: torch.Tensor, spec: ChunkSpec) -> Optional[torch.Tensor]:
        return self._decode(z, trim=spec.is_first_decode)

    # ----- Generic streaming interface (online, arbitrary chunk lengths) ---- #

    @torch.no_grad()
    def encode_chunk(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        ps = self.model.patch_size
        if ps > 1:
            N, T, C, H, W = x.shape
            x = F.pixel_unshuffle(x.reshape(N * T, C, H, W), ps)
            x = x.reshape(N, T, *x.shape[1:])
        if self._enc_left is not None:
            x = torch.cat([self._enc_left, x], dim=1)
            self._enc_left = None
        T = x.shape[1]
        rem = T % 4
        if rem:
            keep = T - rem
            if keep > 0:
                self._enc_left = x[:, keep:].detach().clone()
                x = x[:, :keep]
            else:
                self._enc_left = x.detach().clone()
                return None
        z, self._enc_st = apply_parallel_with_boundary(
            self.model.encoder, x, self._enc_st, self.frame_batch_size, self.fused)
        return z

    @torch.no_grad()
    def flush_encoder(self) -> Optional[torch.Tensor]:
        if self._enc_left is None:
            return None
        x = self._enc_left
        self._enc_left = None
        T = x.shape[1]
        if T % 4:
            p = 4 - T % 4
            x = torch.cat([x, x[:, -1:].expand(-1, p, -1, -1, -1)], dim=1)
        z, self._enc_st = apply_parallel_with_boundary(
            self.model.encoder, x, self._enc_st, self.frame_batch_size, self.fused)
        return z

    @torch.no_grad()
    def decode_chunk(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        x = self._decode(z, trim=self._first_dec)
        self._first_dec = False
        return x

    def _decode(self, z: torch.Tensor, trim: bool) -> Optional[torch.Tensor]:
        x, self._dec_st = apply_parallel_with_boundary(
            self._dec_head, z, self._dec_st, self.frame_batch_size, self.fused)
        if x is None:
            return None
        if trim:
            x = x[:, self.model.frames_to_trim:]
        N, T = x.shape[:2]
        x = run_frame_batches(self._output_frames, x.flatten(0, 1), self.frame_batch_size)
        return x.view(N, T, *x.shape[1:])

    def _output_frames(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(self._dec_tail(x), 0, 1)
        ps = self.model.patch_size
        return F.pixel_shuffle(x, ps) if ps > 1 else x
