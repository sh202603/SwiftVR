"""Spatial tiling for high output resolutions.

The LQ frame is split into overlapping square tiles; each tile is restored over
the whole video into a temporary mp4 (``SwiftVRPipeline.restore_video`` with
``tile_size``), and the tiles are then blended with feather weights into the
final output. Port of FlashVSR_plus' tiny-long tiling, with two differences:
tiles are upscaled with a margin so their input equals the whole-frame upscale,
and feather ramps are only applied on edges shared with a neighbouring tile.

``plan_tiled_run`` does not need the model, so the CLI can call it first and
skip loading the model when every tile of a resumed run is already complete.
"""

import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

import decord

from .io import (
    CROP_LQ_MULTIPLE,
    CROP_LQ_MULTIPLE_PAD_ALIGN,
    UPSCALE_MARGIN,
    VIDEO_EXTS,
    _decord_batch_to_torch,
    append_chunk_to_png_dir,
    get_video_info,
    list_image_frames,
    open_stream_video_writer,
    selected_output_frame_names,
)
from .resume import (
    TEMP_DIR_NAME,
    TILE_DIR_PREFIX,
    build_resume_manifest,
    input_fingerprint,
    list_stale_tile_dirs,
    prepare_resume_dir,
    resume_manifest_hash,
    scan_completed_tiles,
    tile_video_path,
)

TEMP_PIX_FMT = "yuv444p"
# Below this upscaled tile size the shifted-window attention has at most 24
# tokens per side, where the shifted and unshifted window starts coincide.
MIN_SHIFT_EFFECTIVE_TILE_OUT = 1024


def _log(msg, verbose=True):
    if verbose:
        print(f"[swiftvr] {msg}", flush=True)


def _warn(msg):
    print(f"[swiftvr] Warning: {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Tile geometry                                                               #
# --------------------------------------------------------------------------- #

def calculate_tile_coords(height, width, tile_size, overlap) -> List[Tuple[int, int, int, int]]:
    """``(x1, y1, x2, y2)`` per tile, row-major. Edge tiles are shifted inwards
    so every tile is ``tile_size`` square (requires ``tile_size <= min(H, W)``)."""
    coords = []
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    for r in range(num_rows):
        for c in range(num_cols):
            y1, x1 = r * stride, c * stride
            y2, x2 = min(y1 + tile_size, height), min(x1 + tile_size, width)
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
            coords.append((x1, y1, x2, y2))
    return coords


def create_feather_mask(tile_h, tile_w, ramp_len, *, left, right, top, bottom,
                        device=None) -> torch.Tensor:
    """``[tile_h, tile_w]`` float32 weights: 0 -> 1 linear ramps of ``ramp_len``
    px on the edges flagged True (edges shared with a neighbouring tile), 1
    elsewhere. Edges on the frame border get no ramp: FlashVSR_plus ramps them
    too, which gives the outermost pixel row/column a total weight of 0 and
    renders it black."""
    ramp = torch.linspace(0, 1, ramp_len, dtype=torch.float32, device=device)
    wx = torch.ones(tile_w, dtype=torch.float32, device=device)
    wy = torch.ones(tile_h, dtype=torch.float32, device=device)
    if left:
        wx[:ramp_len] *= ramp
    if right:
        wx[-ramp_len:] *= ramp.flip(0)
    if top:
        wy[:ramp_len] *= ramp
    if bottom:
        wy[-ramp_len:] *= ramp.flip(0)
    return wy[:, None] * wx[None, :]


def tile_masks(tile_coords, lq_h, lq_w, upscale, overlap, device=None):
    """Feather mask per tile and their sum (the normalising weight canvas),
    in upscaled pixels."""
    final_h, final_w = lq_h * upscale, lq_w * upscale
    weight = torch.zeros(final_h, final_w, dtype=torch.float32, device=device)
    masks = []
    for x1, y1, x2, y2 in tile_coords:
        m = create_feather_mask((y2 - y1) * upscale, (x2 - x1) * upscale, overlap * upscale,
                                left=x1 > 0, right=x2 < lq_w, top=y1 > 0, bottom=y2 < lq_h,
                                device=device)
        masks.append(m)
        weight[y1 * upscale:y2 * upscale, x1 * upscale:x2 * upscale] += m
    weight[weight == 0] = 1.0
    return masks, weight


# --------------------------------------------------------------------------- #
# Validation and planning                                                     #
# --------------------------------------------------------------------------- #

def _tile_size_candidates(tile_size, upscale, limit):
    step = 32 // math.gcd(32, upscale)
    base = (tile_size // step) * step
    cands = [base + k * step for k in (-1, 0, 1, 2)]
    return [c for c in cands if 0 < c <= limit and c != tile_size]


def validate_tiling(*, tile_size, overlap, upscale, lq_h, lq_w, resolution, upscale_mode):
    if resolution is not None:
        raise ValueError("tile_size cannot be combined with resolution: tiling needs an integer "
                         "upscale factor. Use upscale, and output_height to resize the result.")
    if upscale_mode not in UPSCALE_MARGIN:
        raise ValueError(f"Tiling supports upscale_mode {sorted(UPSCALE_MARGIN)}, got {upscale_mode!r}.")
    if overlap <= 0:
        raise ValueError(f"tile overlap must be positive, got {overlap}.")
    if overlap > tile_size / 2:
        raise ValueError(f"tile overlap ({overlap}) must be at most half of tile_size ({tile_size}).")
    if tile_size > min(lq_h, lq_w):
        raise ValueError(f"tile_size ({tile_size}) must not exceed the smaller input dimension "
                         f"({min(lq_h, lq_w)}); use a smaller tile or disable tiling.")
    if (tile_size * upscale) % 32 != 0:
        cands = _tile_size_candidates(tile_size, upscale, min(lq_h, lq_w))
        hint = f" (for upscale {upscale} use e.g. {', '.join(map(str, cands))})" if cands else ""
        raise ValueError(f"tile_size x upscale ({tile_size} x {upscale}) must be a multiple of 32"
                         f"{hint}.")
    if tile_size * upscale < MIN_SHIFT_EFFECTIVE_TILE_OUT:
        _warn(f"tile_size x upscale = {tile_size * upscale} < {MIN_SHIFT_EFFECTIVE_TILE_OUT}: the "
              "shifted-window attention cannot shift inside such small tiles, which may lower "
              "quality.")


@dataclass
class TilePlan:
    input_path: Path
    total_frames: int
    lq_h: int
    lq_w: int
    fps: float
    upscale: int
    tile_size: int
    overlap: int
    tile_coords: List[Tuple[int, int, int, int]]
    tile_dir: Path
    completed: set
    final_video_path: Path
    png_output_dir: Path
    png_save: bool
    png_frame_names: Optional[List[str]]
    manifest: dict = field(repr=False)

    @property
    def num_tiles(self):
        return len(self.tile_coords)

    @property
    def all_complete(self):
        return len(self.completed) == self.num_tiles

    @property
    def output(self):
        return str(self.png_output_dir if self.png_save else self.final_video_path)


def resolve_output(input_path: Path, output_path: Path, png_save: bool):
    """``(final_video_path, png_output_dir)``; creates the directories."""
    if png_save:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / f"{input_path.stem}.mp4", output_path
    if output_path.suffix.lower() in VIDEO_EXTS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path, output_path.parent
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{input_path.stem}.mp4", output_path


def resolve_attention_backend(name: str) -> str:
    """The backend ``set_attention_backend(name)`` would select, without a model."""
    if name == "auto":
        from .models.transformer import _pick_best_backend
        return _pick_best_backend()
    return name


def plan_tiled_run(
    input_path,
    output_path,
    *,
    tile_size: int,
    tile_overlap: int = 24,
    upscale: int = 4,
    resolution=None,
    clip_len: int = 24,
    dit_overlap: int = 0,
    fps: Optional[float] = None,
    png_save: bool = False,
    pad_align: bool = False,
    upscale_mode: str = "bilinear",
    dtype: str = "torch.bfloat16",
    attention_backend: str = "auto",
    fp8_dit: bool = False,
    torch_compile: bool = False,
    reae_fused: bool = True,
    reae_frame_batch_size: Optional[int] = None,
    cudnn_benchmark: bool = False,
    temp_quality: int = 80,
    resume: bool = False,
    temp_dir=None,
    verbose: bool = True,
) -> TilePlan:
    """Validate the tiling, compute the tile coordinates, prepare the temp tile
    directory and find the tiles a previous run completed. Needs no model.

    ``dtype`` and ``attention_backend`` must be what the pipeline actually runs
    with (``str(torch.dtype)``, a concrete backend or ``"auto"``); they key the
    resume manifest."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    crop_multiple = CROP_LQ_MULTIPLE_PAD_ALIGN if pad_align else CROP_LQ_MULTIPLE
    raw_total, lq_h, lq_w, src_fps = get_video_info(input_path, fallback_fps=fps or 30,
                                                     crop_multiple=crop_multiple)
    total_frames = 4 * ((raw_total - 1) // 4) + 1
    if clip_len % 4 != 0:
        raise ValueError(f"clip_len must be a multiple of 4, got {clip_len}")
    validate_tiling(tile_size=tile_size, overlap=tile_overlap, upscale=upscale, lq_h=lq_h,
                    lq_w=lq_w, resolution=resolution, upscale_mode=upscale_mode)
    tile_coords = calculate_tile_coords(lq_h, lq_w, tile_size, tile_overlap)

    final_video_path, png_output_dir = resolve_output(input_path, output_path, png_save)
    image_frames = list_image_frames(input_path) if input_path.is_dir() else None
    png_frame_names = selected_output_frame_names(input_path) if (png_save and image_frames) else None

    params = {
        "input": input_fingerprint(input_path, image_frames),
        "total_frames": total_frames,
        "lq_h": lq_h,
        "lq_w": lq_w,
        "upscale": upscale,
        "tile_size": tile_size,
        "overlap": tile_overlap,
        "clip_len": clip_len,
        "dit_overlap": dit_overlap,
        "dtype": str(dtype),
        "upscale_mode": upscale_mode,
        "pad_align": bool(pad_align),
        "attention_backend": resolve_attention_backend(attention_backend),
        "fp8_dit": bool(fp8_dit),
        "torch_compile": bool(torch_compile),
        "reae_fused": bool(reae_fused),
        "reae_frame_batch_size": reae_frame_batch_size,
        "cudnn_benchmark": bool(cudnn_benchmark),
        "temp_quality": int(temp_quality),
        "temp_pix_fmt": TEMP_PIX_FMT,
    }
    manifest = build_resume_manifest(params, {"fps": fps or src_fps, "num_tiles": len(tile_coords)})

    # The PNG folder is the product itself, so temp files go next to it.
    temp_root = Path(temp_dir) if temp_dir else (
        (png_output_dir.parent if png_save else final_video_path.parent) / TEMP_DIR_NAME)
    tile_dir = temp_root / (TILE_DIR_PREFIX + resume_manifest_hash(manifest))
    completed = set()
    if prepare_resume_dir(tile_dir, manifest, resume):
        completed = scan_completed_tiles(tile_dir, len(tile_coords), total_frames, verbose)
        _log(f"Resume: {len(completed)}/{len(tile_coords)} tiles already complete, "
             f"{len(tile_coords) - len(completed)} to compute.", verbose)
    stale = list_stale_tile_dirs(temp_root, tile_dir)
    if stale:
        _log(f"{len(stale)} other tile director{'y' if len(stale) == 1 else 'ies'} in {temp_root} "
             f"(earlier or concurrent runs; remove them manually if unused): "
             f"{', '.join(p.name for p in stale)}", verbose)

    return TilePlan(
        input_path=input_path, total_frames=total_frames, lq_h=lq_h, lq_w=lq_w,
        fps=float(fps or src_fps), upscale=upscale, tile_size=tile_size, overlap=tile_overlap,
        tile_coords=tile_coords, tile_dir=tile_dir, completed=completed,
        final_video_path=final_video_path, png_output_dir=png_output_dir, png_save=png_save,
        png_frame_names=png_frame_names, manifest=manifest)


# --------------------------------------------------------------------------- #
# Resource estimates                                                          #
# --------------------------------------------------------------------------- #

def estimate_tile_vram_gib(tile_out_px: int, fp8_dit: bool) -> float:
    """Peak allocated GPU memory (weights included) for one tile of
    ``tile_out_px`` squared output pixels, bf16. Linear fit of measurements on
    an RTX 5060 Ti with the default ReAE settings: 1024/1280/1536 px tiles peak
    at 7.5/9.0/10.8 GiB with FP8, and 4.6 GiB more in bf16."""
    mp = tile_out_px * tile_out_px / 1e6
    return (9.4 if not fp8_dit else 4.9) + 2.53 * mp


def estimate_temp_disk_gib(plan: TilePlan, temp_quality: int) -> float:
    """Size of the tile videos still to compute. Fit of yuv444p x265 bitrates
    measured on one clip (0.15/0.65/1.55 MiB/s per output megapixel at quality
    60/80/90), doubled because the bitrate depends on the content."""
    tile_out_mp = (plan.tile_size * plan.upscale) ** 2 / 1e6
    crf = round((100 - max(0, min(100, temp_quality))) * 51 / 100)
    mib_per_s_per_mp = 2 * 0.15 * 2 ** (0.225 * (20 - crf))
    seconds = plan.total_frames / max(plan.fps, 1.0)
    remaining = plan.num_tiles - len(plan.completed)
    return remaining * seconds * tile_out_mp * mib_per_s_per_mp / 1024


def log_resource_estimates(plan: TilePlan, *, temp_quality, fp8_dit, device, verbose=True):
    tile_out = plan.tile_size * plan.upscale
    final_w, final_h = plan.lq_w * plan.upscale, plan.lq_h * plan.upscale
    _log(f"{plan.num_tiles} tiles of {plan.tile_size}px (output {tile_out}x{tile_out}), "
         f"overlap {plan.overlap}px, final {final_w}x{final_h}, {plan.total_frames} frames.", verbose)
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        # The model is resident when this runs, and memory PyTorch has cached
        # can be reused, so the estimate (weights included) is compared with
        # free + reserved.
        free_b, _ = torch.cuda.mem_get_info(device)
        usable = (free_b + torch.cuda.memory_reserved(device)) / 2 ** 30
        est = estimate_tile_vram_gib(tile_out, fp8_dit)
        _log(f"Estimated peak GPU memory per tile ~{est:.1f} GiB "
             f"(available to PyTorch: {usable:.1f} GiB).", verbose)
        if est > usable * 0.95:
            _warn("the estimated peak is close to or above the free GPU memory; consider a "
                  "smaller tile_size.")
    est_disk = estimate_temp_disk_gib(plan, temp_quality)
    plan.tile_dir.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(plan.tile_dir).free / 2 ** 30
    _log(f"Temp tile videos: ~{est_disk:.1f} GiB estimated, kept in {plan.tile_dir} until "
         f"stitching finishes (free disk: {free_disk:.1f} GiB).", verbose)
    if est_disk > free_disk:
        _warn("free disk space may be insufficient for the temp tile videos.")


# --------------------------------------------------------------------------- #
# Stitching                                                                   #
# --------------------------------------------------------------------------- #

def _output_size(final_h, final_w, output_height, verbose):
    if output_height is None:
        return None
    if output_height >= final_h:
        _warn(f"output_height {output_height} >= native height {final_h}; keeping the native "
              "resolution.")
        return None
    out_h = output_height - output_height % 2
    out_w = int(round(final_w * out_h / final_h / 2)) * 2
    _log(f"Stitching at {final_w}x{final_h}, downscaling the output to {out_w}x{out_h}.", verbose)
    return out_h, out_w


def _stitch_chunk_frames(final_h, final_w, tile_out, device, requested=None):
    """Frames per stitching chunk from the free memory: the float32 canvas, one
    float32 tile and the uint8 result per frame, with half the budget spare."""
    per_frame = final_h * final_w * 3 * (4 + 1) + tile_out * tile_out * 3 * 4
    if device.type == "cuda":
        free_b, _ = torch.cuda.mem_get_info(device)
        budget = free_b // 2
    else:
        budget = 4 * 2 ** 30
    n = max(1, int(budget // per_frame))
    return min(n, requested or 32)


def stitch_tiles(plan: TilePlan, *, quality=85, save_format="", ffmpeg_preset="",
                 output_height=None, device="cuda", chunk_frames=None, verbose=True) -> int:
    """Blend the tile videos into the final mp4 or PNG sequence; returns the
    number of frames written. Blending runs in float32 on ``device``."""
    from .runner import _Progress

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    up = plan.upscale
    final_h, final_w = plan.lq_h * up, plan.lq_w * up
    tile_out = plan.tile_size * up
    total = plan.total_frames

    masks, weight = tile_masks(plan.tile_coords, plan.lq_h, plan.lq_w, up, plan.overlap, device)
    inv_weight = (1.0 / weight)[None, :, :, None]
    del weight
    out_size = _output_size(final_h, final_w, output_height, verbose)
    n_chunk = _stitch_chunk_frames(final_h, final_w, tile_out, device, chunk_frames)

    readers = []
    vr = None
    writer = None
    written = 0
    png_written_once = set()
    try:
        for i in range(plan.num_tiles):
            vr = decord.VideoReader(tile_video_path(plan.tile_dir, i))
            if len(vr) != total:
                raise RuntimeError(f"Tile video {i + 1} has {len(vr)} frames, expected {total}; "
                                   "the tiled run is incomplete, refusing to write a broken output.")
            readers.append(vr)

        progress = _Progress(total, verbose, "stitch ")
        for start in range(0, total, n_chunk):
            n = min(n_chunk, total - start)
            t0 = time.perf_counter()
            canvas = torch.zeros(n, final_h, final_w, 3, dtype=torch.float32, device=device)
            for i, vr in enumerate(readers):
                frames = _decord_batch_to_torch(vr.get_batch(list(range(start, start + n))))
                if frames.shape[0] != n:
                    raise RuntimeError(f"Tile video {i + 1} ended early at frame {start + frames.shape[0]}"
                                       f"/{total}; refusing to write a broken output.")
                if tuple(frames.shape[1:3]) != (tile_out, tile_out):
                    raise RuntimeError(f"Tile video {i + 1} is {frames.shape[2]}x{frames.shape[1]}, "
                                       f"expected {tile_out}x{tile_out}.")
                x1, y1, _, _ = plan.tile_coords[i]
                oy, ox = y1 * up, x1 * up
                tile = frames.to(device, non_blocking=True).to(torch.float32)
                canvas[:, oy:oy + tile_out, ox:ox + tile_out, :].addcmul_(tile, masks[i][None, :, :, None])
                del tile, frames
            canvas.mul_(inv_weight)
            if out_size is not None:
                canvas = F.interpolate(canvas.permute(0, 3, 1, 2), size=out_size, mode="area").permute(0, 2, 3, 1)
            # Round, not truncate: truncation darkens blended pixels by 0.5 LSB on average.
            out = canvas.round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
            del canvas

            if plan.png_save:
                saved, _ = append_chunk_to_png_dir(out, str(plan.png_output_dir), start_idx=start,
                                                   frame_names=plan.png_frame_names,
                                                   written_once=png_written_once)
                written += saved
            else:
                if writer is None:
                    writer = open_stream_video_writer(str(plan.final_video_path), fps=plan.fps,
                                                      video_format=save_format, preset=ffmpeg_preset,
                                                      quality=quality)
                for frame in out:
                    writer.append_data(frame)
                written += n
            dt = time.perf_counter() - t0
            progress.update(start + n, n / dt if dt > 0 else 0.0)
        progress.close()
    finally:
        if writer is not None:
            writer.close()
        # decord has no close(); the files stay open until the readers are freed,
        # and Windows cannot delete open files.
        vr = None
        readers.clear()
    return written


def finish_tiled_run(plan: TilePlan, *, quality=85, save_format="", ffmpeg_preset="",
                     output_height=None, device="cuda", verbose=True, t_start=None,
                     peak_allocated=0, peak_reserved=0) -> dict:
    """Stitch, delete the temp tile directory and return ``restore_video``-style
    stats. ``t_start`` and the peaks carry over from the tile loop, if any."""
    t_start = time.perf_counter() if t_start is None else t_start
    dev = torch.device(device)
    use_cuda = dev.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    _log(f"Stitching {plan.num_tiles} tiles -> {plan.output}", verbose)
    written = stitch_tiles(plan, quality=quality, save_format=save_format, ffmpeg_preset=ffmpeg_preset,
                           output_height=output_height, device=dev, verbose=verbose)
    if use_cuda:
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(dev))
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(dev))
    shutil.rmtree(plan.tile_dir)
    if plan.tile_dir.parent.name == TEMP_DIR_NAME:
        try:
            os.rmdir(plan.tile_dir.parent)  # only succeeds when empty
        except OSError:
            pass
    wall = time.perf_counter() - t_start
    return {"frames": written, "seconds": wall,
            "fps": (written / wall if wall > 0 else 0.0),
            "output": plan.output,
            "max_memory_allocated": peak_allocated,
            "max_memory_reserved": peak_reserved}
