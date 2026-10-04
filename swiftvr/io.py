"""Input/output utilities for SwiftVR: frame reading, GPU preprocessing and writing.

Supports both video files (probed with ``decord``, decoded by an ffmpeg
subprocess) and image folders, and writes
either an mp4 (libx265) or a PNG sequence.
"""

import os
import re
import math
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import imageio
import imageio_ffmpeg
from PIL import Image

import decord
decord.bridge.set_bridge("torch")

from .streaming.chunk import ChunkSpec, build_chunk_specs

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
KEEP_MAX_4K1 = True
CROP_LQ_MULTIPLE = 8
# With ``pad_align`` the LQ input is only cropped to even sizes (yuv420p mp4
# needs even dimensions); the 32-alignment is then done by reflect padding.
CROP_LQ_MULTIPLE_PAD_ALIGN = 2

# LQ pixels that must surround a crop so that upscaling the crop and dropping
# the margin equals upscaling the whole frame and cropping (integer factors,
# ``align_corners=False``): bilinear reads the 1 px neighbourhood, bicubic 2 px.
UPSCALE_MARGIN = {"bilinear": 1, "bicubic": 2, "nearest": 0}

_INTERP_NEEDS_ALIGN = ("linear", "bilinear", "bicubic", "trilinear")


def is_video_file(filename) -> bool:
    return str(filename).lower().endswith(VIDEO_EXTS)


# --------------------------------------------------------------------------- #
# Frame listing / reading                                                     #
# --------------------------------------------------------------------------- #

def _is_valid_image_file(path: Path) -> bool:
    if not path.is_file() or path.name.startswith("."):
        return False
    return path.suffix.lower() in IMAGE_EXTS


def _numeric_sort_key(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return path.name


def _max_4k_plus_1_count(n: int) -> int:
    return ((n - 1) // 4) * 4 + 1 if n > 0 else 0


def list_image_frames(folder_path) -> List[Path]:
    folder = Path(folder_path)
    frames = sorted((p for p in folder.iterdir() if _is_valid_image_file(p)), key=_numeric_sort_key)
    if KEEP_MAX_4K1:
        frames = frames[:_max_4k_plus_1_count(len(frames))]
    return frames


def _crop_size_to_multiple(h: int, w: int, multiple: int) -> Tuple[int, int]:
    if multiple is None or multiple <= 1:
        return h, w
    crop_h, crop_w = (h // multiple) * multiple, (w // multiple) * multiple
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"Invalid crop: {h}x{w}, multiple={multiple}")
    return crop_h, crop_w


class FfmpegFrameReader:
    """Sequential RGB frames of a video file, decoded by an ffmpeg subprocess.

    Not decord: its reader keeps decoding ahead into RAM while the caller is
    busy, ~600-7000 frames per reader depending on the file (3.7 GiB for a
    720p HEVC input), and the eight tile readers of a stitch exhausted memory.
    The pipe to ffmpeg bounds the read-ahead. Rotation metadata is ignored as
    in decord; the frames and the frame count are identical to decord's."""

    def __init__(self, path):
        self._gen = imageio_ffmpeg.read_frames(
            str(path), pix_fmt="rgb24", input_params=["-noautorotate"],
            output_params=["-fps_mode", "passthrough", "-an", "-sn"])
        self.width, self.height = next(self._gen)["size"]

    def read(self, n: int) -> torch.Tensor:
        """Up to ``n`` frames as a CPU ``[n, H, W, 3]`` uint8 tensor; fewer at the end."""
        frames = []
        for _ in range(n):
            try:
                buf = next(self._gen)
            except StopIteration:
                break
            frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3))
        if not frames:
            return torch.empty(0, self.height, self.width, 3, dtype=torch.uint8)
        return torch.from_numpy(np.stack(frames))

    def close(self):
        """Stops ffmpeg; Windows cannot delete a file it holds open."""
        self._gen.close()


def _read_image_chunk_uint8(paths: List[Path], crop_rect: Tuple[int, int, int, int]) -> torch.Tensor:
    x1, y1, x2, y2 = crop_rect
    arrs = []
    for p in paths:
        with Image.open(p) as img:
            img = img.convert("RGB")
            w, h = img.size
            if h < y2 or w < x2:
                raise ValueError(f"Frame too small: {p}, frame={h}x{w}, crop={crop_rect}")
            arrs.append(np.asarray(img.crop((x1, y1, x2, y2)), dtype=np.uint8))
    return torch.from_numpy(np.stack(arrs, axis=0)).contiguous()


def get_video_info(video_path, fallback_fps=30,
                   crop_multiple: int = CROP_LQ_MULTIPLE) -> Tuple[int, int, int, float]:
    """Return (total_frames, lq_height, lq_width, fps) with sizes cropped to a
    multiple of ``crop_multiple``."""
    path = Path(video_path)
    if path.is_dir():
        frames = list_image_frames(path)
        if not frames:
            raise ValueError(f"No valid image frames in: {path}")
        with Image.open(frames[0]) as img:
            w, h = img.convert("RGB").size
        crop_h, crop_w = _crop_size_to_multiple(h, w, crop_multiple)
        return len(frames), crop_h, crop_w, float(fallback_fps)

    vr = decord.VideoReader(uri=path.as_posix())
    f0 = vr[0]
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = float(fallback_fps)
    if not math.isfinite(fps) or fps <= 0:
        fps = float(fallback_fps)
    crop_h, crop_w = _crop_size_to_multiple(f0.shape[0], f0.shape[1], crop_multiple)
    return len(vr), crop_h, crop_w, fps


def selected_output_frame_names(input_folder) -> List[str]:
    return [f"{p.stem}.png" for p in list_image_frames(Path(input_folder))]


def margin_crop_rect(rect: Tuple[int, int, int, int], margin: int, lq_h: int, lq_w: int,
                     upscale: int) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Grow the LQ crop ``rect`` = ``(x1, y1, x2, y2)`` by ``margin`` px on the
    sides that are inside the ``lq_w`` x ``lq_h`` frame. Returns the read
    rectangle and the ``(top, bottom, left, right)`` margin in upscaled pixels
    that ``preprocess_clip_uint8`` trims after resizing."""
    x1, y1, x2, y2 = rect
    rx1, ry1 = max(0, x1 - margin), max(0, y1 - margin)
    rx2, ry2 = min(lq_w, x2 + margin), min(lq_h, y2 + margin)
    trim = ((y1 - ry1) * upscale, (ry2 - y2) * upscale, (x1 - rx1) * upscale, (rx2 - x2) * upscale)
    return (rx1, ry1, rx2, ry2), trim


def iter_video_clips_fixed_scheme(
    video_path, clip_len: int, total_frames: int, crop_rect: Tuple[int, int, int, int],
) -> Iterator[Tuple[ChunkSpec, torch.Tensor]]:
    """Yield ``(spec, raw_uint8_frames)`` for each fixed-size chunk.

    ``crop_rect`` is ``(x1, y1, x2, y2)`` in source pixels; ``(0, 0, lq_w, lq_h)``
    reads the whole (cropped) frame. ``raw_uint8_frames`` is a CPU
    ``[T, y2-y1, x2-x1, 3]`` uint8 tensor.
    """
    x1, y1, x2, y2 = crop_rect
    path = Path(video_path)
    specs = build_chunk_specs(total_frames, clip_len)

    if path.is_dir():
        all_paths = list_image_frames(path)
        for spec in specs:
            chunk_paths = all_paths[spec.frame_start: spec.frame_start + spec.frame_count]
            yield spec, _read_image_chunk_uint8(chunk_paths, crop_rect)
    else:
        # The chunks are consecutive, so a sequential reader suffices.
        reader = FfmpegFrameReader(path)
        try:
            for spec in specs:
                frames = reader.read(spec.frame_count)
                if frames.shape[0] != spec.frame_count:
                    raise RuntimeError(f"{path} ended at frame {spec.frame_start + frames.shape[0]}, "
                                       f"expected {total_frames} frames.")
                yield spec, frames[:, y1:y2, x1:x2, :].contiguous()
        finally:
            reader.close()


# --------------------------------------------------------------------------- #
# GPU preprocessing                                                           #
# --------------------------------------------------------------------------- #

def preprocess_clip_uint8(frames_uint8, out_h, out_w, mode, pad_h, pad_w, dtype,
                          trim=None, pad_mode="constant"):
    """``[T, H, W, 3]`` uint8 (CUDA) -> ``[1, T, 3, out_h+pad_h, out_w+pad_w]``
    float in [0, 1]. Resizing and padding run on the GPU.

    ``trim`` = ``(top, bottom, left, right)``: the frames carry a margin that is
    resized to ``out + trim`` and then dropped (see ``margin_crop_rect``).
    ``pad_mode`` is ``"constant"`` (zeros) or ``"reflect"``; reflect falls back
    to replicate when the padding is not smaller than the frame."""
    frames = frames_uint8.permute(0, 3, 1, 2).contiguous().to(dtype=dtype)
    _, _, h, w = frames.shape
    top, bottom, left, right = trim or (0, 0, 0, 0)
    rs_h, rs_w = out_h + top + bottom, out_w + left + right
    if (h, w) != (rs_h, rs_w):
        if mode in _INTERP_NEEDS_ALIGN:
            frames = F.interpolate(frames, size=(rs_h, rs_w), mode=mode, align_corners=False)
        else:
            frames = F.interpolate(frames, size=(rs_h, rs_w), mode=mode)
    if top or bottom or left or right:
        frames = frames[:, :, top:top + out_h, left:left + out_w]
    frames = frames / 255.0
    if pad_h > 0 or pad_w > 0:
        if pad_mode == "constant":
            frames = F.pad(frames, (0, pad_w, 0, pad_h), mode="constant", value=0)
        else:
            if pad_mode == "reflect" and (pad_h >= out_h or pad_w >= out_w):
                pad_mode = "replicate"
            frames = F.pad(frames, (0, pad_w, 0, pad_h), mode=pad_mode)
    return frames.unsqueeze(0)


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #

def crop_spatial_padding_ntchw(video, pad_h=0, pad_w=0):
    if video is None:
        return None
    if pad_h > 0:
        video = video[:, :, :, :-pad_h, :]
    if pad_w > 0:
        video = video[:, :, :, :, :-pad_w]
    return video


def ntchw_to_uint8_thwc(video):
    """``[1, T, 3, H, W]`` float in [0, 1] -> ``[T, H, W, 3]`` uint8 on the same
    device. Converting on the GPU halves the D2H payload vs bf16."""
    if video is None or video.numel() == 0 or video.shape[1] == 0:
        return None
    return (video[0].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).contiguous()


def quality_to_crf(quality: int) -> int:
    q = max(0, min(100, int(quality)))
    return int(round((100 - q) * 51 / 100))


def open_stream_video_writer(output_path, fps=8, video_format="", preset="", quality=85):
    extra = ["-preset", str(preset)] if preset else []
    crf = quality_to_crf(quality)
    pix = "yuv444p" if video_format == "yuv444p" else "yuv420p"
    # libx265 logs through its own logger (not ffmpeg's -loglevel); silence its
    # banner and final statistics.
    return imageio.get_writer(output_path, fps=fps, codec="libx265", pixelformat=pix,
                             macro_block_size=None, ffmpeg_log_level="error",
                             ffmpeg_params=["-crf", str(crf), "-x265-params", "log-level=error"] + extra)


_AUDIO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?: Audio:")


def has_audio_stream(video_path) -> bool:
    """True if ``video_path`` contains at least one audio stream.

    imageio-ffmpeg ships ffmpeg but not ffprobe, so the stream list is read
    from ``ffmpeg -i`` (which exits non-zero for want of an output file; the
    stream info is still printed to stderr)."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(video_path)]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE)
    return _AUDIO_STREAM_RE.search(proc.stderr.decode("utf-8", errors="replace")) is not None


def copy_audio_tracks(video_path, audio_source_path, verbose=True) -> bool:
    """Stream-copy all audio tracks of ``audio_source_path`` into the silent
    ``video_path`` (the mp4 written by the pipeline), in place; the video
    stream is copied, not re-encoded. Returns True if audio was added.

    Nothing happens when the source is an image folder or has no audio. If
    the mux fails (e.g. an audio codec the output container cannot hold) the
    silent video is kept and a warning is printed, so a long restoration run
    never ends in a missing output."""
    video_path = Path(video_path)
    audio_source_path = Path(audio_source_path)
    if audio_source_path.is_dir() or not has_audio_stream(audio_source_path):
        return False

    if verbose:
        print("[swiftvr] Copying audio tracks...", flush=True)
    # The silent video is moved aside and ffmpeg writes the muxed file to the
    # final path; ``os.replace`` also overwrites a leftover from a crashed run.
    temp = video_path.with_name(video_path.name + ".noaudio" + video_path.suffix)
    os.replace(video_path, temp)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
           "-i", str(temp), "-i", str(audio_source_path),
           "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy", str(video_path)]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE)
    if proc.returncode == 0:
        os.remove(temp)
        return True
    err = proc.stderr.decode("utf-8", errors="replace").strip()
    print(f"[swiftvr] Warning: audio merge failed; the output has no audio.\n{err}", flush=True)
    if video_path.exists():
        os.remove(video_path)
    os.replace(temp, video_path)
    return False


def _normalize_png_name(name: str) -> str:
    name = str(name)
    return name if name.lower().endswith(".png") else f"{name}.png"


def append_chunk_to_png_dir(frames, output_dir, start_idx=0,
                            frame_names: Optional[List[str]] = None,
                            written_once: Optional[set] = None):
    """Write ``frames`` (``[T, H, W, 3]`` uint8 numpy) as PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    if frames is None or len(frames) == 0:
        return 0, 0

    saved = 0
    for i, frame in enumerate(frames):
        idx = int(start_idx + i)
        if frame_names is not None:
            if idx >= len(frame_names):
                continue
            file_name = _normalize_png_name(frame_names[idx])
        else:
            file_name = f"{idx:05d}.png"
        out_path = os.path.join(output_dir, file_name)
        key = os.path.abspath(out_path)
        if written_once is not None and key in written_once:
            continue
        Image.fromarray(frame).save(out_path, format="PNG", compress_level=0, optimize=False)
        if written_once is not None:
            written_once.add(key)
        saved += 1
    return saved, int(frames.shape[0])
