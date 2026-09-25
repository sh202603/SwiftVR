"""High-level SwiftVR pipeline.

    from swiftvr import SwiftVRPipeline

    pipe = SwiftVRPipeline.from_pretrained("checkpoints/").to("cuda", dtype="bfloat16")
    pipe.restore_video("low_quality.mp4", "restored.mp4", resolution=(1920, 1080))

The pipeline wraps the autoencoder, the one-step DiT and an empty text prompt
embedding, and exposes both an offline whole-file API (``restore_video``) and a
causal chunk-by-chunk API (``stream``).
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from safetensors.torch import load_file

from .models import ReAE, WanTransformer3DModel
from .streaming import StreamingTAE, StreamingDiT
from .streaming.tae import to_channels_last
from .io import (
    get_video_info,
    selected_output_frame_names,
    preprocess_clip_uint8,
    crop_spatial_padding_ntchw,
    margin_crop_rect,
    CROP_LQ_MULTIPLE,
    CROP_LQ_MULTIPLE_PAD_ALIGN,
    UPSCALE_MARGIN,
)
from .runner import run_pipeline, enable_max_fps_runtime
from .resume import tile_video_path, write_done_marker
from . import tiling


_DTYPES = {"float16": torch.float16, "fp16": torch.float16,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
           "float32": torch.float32, "fp32": torch.float32}


def _as_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}. Choose float16 / bfloat16 / float32.")
    return _DTYPES[key]


# Frames per batch for the ReAE's stateless layers; ``None`` runs a whole chunk
# at once (fastest, highest peak memory).
DEFAULT_REAE_FRAME_BATCH_SIZE = 2


def _aligned_pad(size: int, multiple: int = 32) -> int:
    return (multiple - size % multiple) % multiple


class SwiftVRPipeline:
    def __init__(self, reae, transformer, prompt_emb, upscale_mode: str = "bilinear",
                 reae_frame_batch_size: Optional[int] = DEFAULT_REAE_FRAME_BATCH_SIZE,
                 reae_fused: bool = True):
        self.reae = reae
        self.transformer = transformer
        self.prompt_emb = prompt_emb
        self.upscale_mode = upscale_mode

        # fused: channels-last activations + cuDNN fused conv epilogues (CUDA only)
        self.tae_stream = StreamingTAE(reae, frame_batch_size=reae_frame_batch_size, fused=reae_fused)
        self.dit_stream = StreamingDiT(transformer, overlap=0)

        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._prepared = False
        # Settings fixed by the preparing ``to()`` call; they change the output
        # and so key the tiled-run resume manifest.
        self._fp8_dit = False
        self._torch_compile = False
        self._cudnn_benchmark = False

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir,
        *,
        reae_filename: str = "reae.safetensors",
        transformer_subfolder: str = "transformer",
        prompt_embedding_filename: str = "prompt_embedding.safetensors",
        upscale_mode: str = "bilinear",
        reae_frame_batch_size: Optional[int] = DEFAULT_REAE_FRAME_BATCH_SIZE,
        reae_fused: bool = True,
        device=None,
        dtype=None,
    ) -> "SwiftVRPipeline":
        root = Path(checkpoint_dir)

        reae = ReAE(str(root / reae_filename))
        transformer = WanTransformer3DModel.from_pretrained(str(root), subfolder=transformer_subfolder)
        prompt_emb = load_file(str(root / prompt_embedding_filename))["prompt_emb"][0]

        pipe = cls(reae, transformer, prompt_emb, upscale_mode=upscale_mode,
                   reae_frame_batch_size=reae_frame_batch_size, reae_fused=reae_fused)
        if device is not None or dtype is not None:
            pipe.to(device or "cpu", dtype=dtype or "float32")
        return pipe

    def to(self, device=None, dtype=None, *, attention_backend="auto", torch_compile=False,
           cudnn_benchmark=False, fp8_dit=False):
        """Move the models to ``device``/``dtype`` and prepare them for inference
        (fused projections + shifted-window self-attention, once).

        ``fp8_dit`` runs the DiT block GEMMs in FP8 (bf16 on an sm89+ GPU only):
        several times faster GEMMs and ~4.6 GiB less weight memory, at a small
        accuracy cost. Only takes effect on the first (preparing) call."""
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = _as_dtype(dtype)
        if fp8_dit and not self._prepared:
            from .models.fp8 import fp8_supported
            if self.dtype != torch.bfloat16 or not fp8_supported(self.device):
                raise ValueError("fp8_dit requires dtype=bfloat16 on a CUDA GPU with compute "
                                 "capability 8.9+ (RTX 40 series or newer)")

        self.reae.to(self.device, self.dtype).eval()
        if self.tae_stream.fused and self.device.type == "cuda":
            to_channels_last(self.reae)
        self.transformer.to(self.device, self.dtype).eval()

        enable_max_fps_runtime(allow_tf32=True, cudnn_benchmark=cudnn_benchmark)
        self._cudnn_benchmark = bool(cudnn_benchmark)
        if not self._prepared and hasattr(self.transformer, "prepare_for_inference"):
            self.transformer.prepare_for_inference(
                attention_backend=attention_backend,
                use_torch_compile=torch_compile,
                compile_mode="default",
                fp8=fp8_dit)
            self._prepared = True
            self._fp8_dit = bool(fp8_dit)
            self._torch_compile = bool(torch_compile)
            if fp8_dit:
                torch.cuda.empty_cache()  # return the freed bf16 weights
        return self

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _target_size(self, lq_h, lq_w, resolution, upscale):
        if resolution is not None:
            out_w, out_h = int(resolution[0]), int(resolution[1])
        else:
            out_h, out_w = lq_h * upscale, lq_w * upscale
        return out_h, out_w, _aligned_pad(out_h), _aligned_pad(out_w)

    @staticmethod
    def _resolve_output(input_path: Path, output_path: Path, png_save: bool):
        return tiling.resolve_output(input_path, output_path, png_save)

    # ------------------------------------------------------------------ #
    # Offline (whole file)                                               #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def restore_video(
        self,
        input_path,
        output_path,
        *,
        resolution: Optional[Tuple[int, int]] = None,
        upscale: int = 4,
        clip_len: int = 24,
        dit_overlap: int = 0,
        fps: Optional[float] = None,
        quality: int = 85,
        png_save: bool = False,
        save_format: str = "",
        ffmpeg_preset: str = "",
        queue_size: int = 3,
        verbose: bool = True,
        pad_align: bool = False,
        tile_size: Optional[int] = None,
        tile_overlap: int = 24,
        output_height: Optional[int] = None,
        temp_quality: int = 80,
        resume: bool = False,
        temp_dir=None,
    ) -> dict:
        """Restore a whole video file or image folder.

        ``resolution`` is the output ``(width, height)``; if omitted the output
        is the low-quality input upscaled by ``upscale``. ``clip_len`` must be a
        multiple of 4.

        ``pad_align`` keeps the whole frame: the input is cropped to even sizes
        instead of multiples of 8, and the upscaled frame is reflect-padded
        (instead of zero-padded) to a multiple of 32.

        ``tile_size`` (LQ pixels) restores the video in overlapping square tiles
        of that size, each written to a temporary mp4 (``temp_quality``, in
        ``temp_dir`` or ``_swiftvr_temp`` next to the output) and blended at the
        end; ``tile_overlap`` is in LQ pixels, ``output_height`` downscales the
        blended result, and ``resume`` reuses the tiles that an interrupted run
        with the same settings completed. These only apply with ``tile_size``.
        """
        if clip_len % 4 != 0:
            raise ValueError(f"clip_len must be a multiple of 4, got {clip_len}")
        if not self._prepared:
            self.to()

        if tile_size is not None:
            return self._restore_video_tiled(
                input_path, output_path, resolution=resolution, upscale=upscale, clip_len=clip_len,
                dit_overlap=dit_overlap, fps=fps, quality=quality, png_save=png_save,
                save_format=save_format, ffmpeg_preset=ffmpeg_preset, queue_size=queue_size,
                verbose=verbose, pad_align=pad_align, tile_size=tile_size,
                tile_overlap=tile_overlap, output_height=output_height,
                temp_quality=temp_quality, resume=resume, temp_dir=temp_dir)
        if resume:
            tiling._warn("resume only applies to tiled runs (tile_size); running normally.")
        if output_height is not None:
            tiling._warn("output_height only applies to tiled runs (tile_size); ignoring it.")

        input_path = Path(input_path)
        output_path = Path(output_path)

        crop_multiple = CROP_LQ_MULTIPLE_PAD_ALIGN if pad_align else CROP_LQ_MULTIPLE
        raw_total, lq_h, lq_w, src_fps = get_video_info(input_path, fallback_fps=fps or 30,
                                                         crop_multiple=crop_multiple)
        total_frames = 4 * ((raw_total - 1) // 4) + 1

        out_h, out_w, pad_h, pad_w = self._target_size(lq_h, lq_w, resolution, upscale)
        final_video_path, png_output_dir = self._resolve_output(input_path, output_path, png_save)
        png_frame_names = (selected_output_frame_names(input_path)
                           if (png_save and input_path.is_dir()) else None)

        self.dit_stream.overlap = dit_overlap

        use_cuda = torch.cuda.is_available() and self.device.type == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

        written, wall = run_pipeline(
            video_path=input_path,
            final_output_path=str(final_video_path),
            png_output_dir=str(png_output_dir),
            tae_stream=self.tae_stream,
            dit_stream=self.dit_stream,
            prompt_emb=self.prompt_emb,
            device=self.device,
            dtype=self.dtype,
            total_frames=total_frames,
            clip_len=clip_len,
            lq_h=lq_h, lq_w=lq_w,
            out_h=out_h, out_w=out_w, pad_h=pad_h, pad_w=pad_w,
            upscale_mode=self.upscale_mode,
            source_fps=(fps or src_fps),
            png_save=png_save,
            quality=quality,
            save_format=save_format,
            ffmpeg_preset=ffmpeg_preset,
            queue_size=queue_size,
            png_frame_names=png_frame_names,
            verbose=verbose,
            pad_mode="reflect" if pad_align else "constant",
        )
        # Peak over this call only (model weights included, since they stay resident).
        max_alloc = torch.cuda.max_memory_allocated(self.device) if use_cuda else 0
        max_reserved = torch.cuda.max_memory_reserved(self.device) if use_cuda else 0
        return {"frames": written, "seconds": wall,
                "fps": (written / wall if wall > 0 else 0.0),
                "output": str(png_output_dir if png_save else final_video_path),
                "max_memory_allocated": max_alloc,
                "max_memory_reserved": max_reserved}

    def _restore_video_tiled(self, input_path, output_path, *, resolution, upscale, clip_len,
                             dit_overlap, fps, quality, png_save, save_format, ffmpeg_preset,
                             queue_size, verbose, pad_align, tile_size, tile_overlap,
                             output_height, temp_quality, resume, temp_dir) -> dict:
        from .models.transformer import get_attention_backend

        t_start = time.perf_counter()
        plan = tiling.plan_tiled_run(
            input_path, output_path, tile_size=tile_size, tile_overlap=tile_overlap,
            upscale=upscale, resolution=resolution, clip_len=clip_len, dit_overlap=dit_overlap,
            fps=fps, png_save=png_save, pad_align=pad_align, upscale_mode=self.upscale_mode,
            dtype=str(self.dtype), attention_backend=get_attention_backend(),
            fp8_dit=self._fp8_dit, torch_compile=self._torch_compile,
            reae_fused=self.tae_stream.fused,
            reae_frame_batch_size=self.tae_stream.frame_batch_size,
            cudnn_benchmark=self._cudnn_benchmark, temp_quality=temp_quality, resume=resume,
            temp_dir=temp_dir, verbose=verbose)
        tiling.log_resource_estimates(plan, temp_quality=temp_quality, fp8_dit=self._fp8_dit,
                                      device=self.device, verbose=verbose)

        self.dit_stream.overlap = dit_overlap
        use_cuda = torch.cuda.is_available() and self.device.type == "cuda"
        peak_alloc = peak_reserved = 0
        tile_out = tile_size * upscale
        margin = UPSCALE_MARGIN[self.upscale_mode]
        gib = 1024 ** 3

        for i, rect in enumerate(plan.tile_coords):
            tag = f"tile {i + 1}/{plan.num_tiles}"
            if i in plan.completed:
                tiling._log(f"Skipping {tag}: already complete (resume).", verbose)
                continue
            tiling._log(f"Processing {tag}: ({rect[0]},{rect[1]}) to ({rect[2]},{rect[3]})", verbose)
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
            # Read the tile with a margin so its upscale equals the whole-frame one.
            read_rect, trim = margin_crop_rect(rect, margin, plan.lq_h, plan.lq_w, upscale)
            written, _ = run_pipeline(
                video_path=plan.input_path,
                final_output_path=tile_video_path(plan.tile_dir, i),
                png_output_dir=str(plan.tile_dir),
                tae_stream=self.tae_stream,
                dit_stream=self.dit_stream,
                prompt_emb=self.prompt_emb,
                device=self.device,
                dtype=self.dtype,
                total_frames=plan.total_frames,
                clip_len=clip_len,
                lq_h=plan.lq_h, lq_w=plan.lq_w,
                out_h=tile_out, out_w=tile_out, pad_h=0, pad_w=0,
                upscale_mode=self.upscale_mode,
                source_fps=plan.fps,
                png_save=False,
                quality=temp_quality,
                save_format=tiling.TEMP_PIX_FMT,
                queue_size=queue_size,
                verbose=verbose,
                crop_rect=read_rect,
                trim=trim,
                progress_prefix=f"{tag} ",
            )
            if written != plan.total_frames:
                raise RuntimeError(f"{tag} wrote {written} frames, expected {plan.total_frames}.")
            # Written on every run, not only resumed ones: the run that crashes
            # is usually the one started without resume.
            write_done_marker(plan.tile_dir, i, written)
            if use_cuda:
                a = torch.cuda.max_memory_allocated(self.device)
                r = torch.cuda.max_memory_reserved(self.device)
                peak_alloc, peak_reserved = max(peak_alloc, a), max(peak_reserved, r)
                tiling._log(f"{tag} peak GPU memory: allocated {a / gib:.2f} GiB, "
                            f"reserved {r / gib:.2f} GiB", verbose)

        return tiling.finish_tiled_run(
            plan, quality=quality, save_format=save_format, ffmpeg_preset=ffmpeg_preset,
            output_height=output_height, device=self.device, verbose=verbose, t_start=t_start,
            peak_allocated=peak_alloc, peak_reserved=peak_reserved)

    # ------------------------------------------------------------------ #
    # Streaming (chunk by chunk, causal)                                 #
    # ------------------------------------------------------------------ #

    def stream(self, *, clip_len: int = 24, resolution: Optional[Tuple[int, int]] = None,
               upscale: int = 4, dit_overlap: int = 1) -> "StreamSession":
        if clip_len % 4 != 0:
            raise ValueError(f"clip_len must be a multiple of 4, got {clip_len}")
        if not self._prepared:
            self.to()
        return StreamSession(self, clip_len=clip_len, resolution=resolution,
                             upscale=upscale, dit_overlap=dit_overlap)


class StreamSession:
    """Causal chunk-by-chunk session. Call ``step`` with each new clip of frames
    and ``flush`` once at the end. Output sizing is taken from ``resolution`` or
    inferred from the first clip via ``upscale``."""

    def __init__(self, pipe: SwiftVRPipeline, clip_len, resolution, upscale, dit_overlap):
        self.pipe = pipe
        self.clip_len = clip_len
        self.resolution = resolution
        self.upscale = upscale
        self._sizes = None

        pipe.tae_stream.reset()
        pipe.dit_stream.reset()
        pipe.dit_stream.overlap = dit_overlap

    def _ensure_sizes(self, lq_h, lq_w):
        if self._sizes is None:
            self._sizes = self.pipe._target_size(lq_h, lq_w, self.resolution, self.upscale)
        return self._sizes

    def _run_latents(self, z):
        z_bcfhw = z.permute(0, 2, 1, 3, 4).contiguous()
        den = self.pipe.dit_stream.denoise(z_bcfhw, self.pipe.prompt_emb)
        return den.permute(0, 2, 1, 3, 4).contiguous()

    @torch.inference_mode()
    def step(self, frames_uint8: torch.Tensor) -> Optional[torch.Tensor]:
        """``frames_uint8``: ``[T, H, W, 3]`` uint8. Returns ``[1, T', 3, H', W']``
        in [0, 1], or ``None`` if the frames were buffered (T not a multiple of 4)."""
        g = frames_uint8.to(self.pipe.device)
        out_h, out_w, pad_h, pad_w = self._ensure_sizes(g.shape[1], g.shape[2])
        clip = preprocess_clip_uint8(g, out_h, out_w, self.pipe.upscale_mode, pad_h, pad_w, self.pipe.dtype)

        z = self.pipe.tae_stream.encode_chunk(clip)
        if z is None:
            return None
        z_ntchw = self._run_latents(z)
        rgb = self.pipe.tae_stream.decode_chunk(z_ntchw)
        return crop_spatial_padding_ntchw(rgb, pad_h, pad_w)

    @torch.inference_mode()
    def flush(self) -> Optional[torch.Tensor]:
        z = self.pipe.tae_stream.flush_encoder()
        if z is None or self._sizes is None:
            return None
        _, _, pad_h, pad_w = self._sizes
        z_ntchw = self._run_latents(z)
        rgb = self.pipe.tae_stream.decode_chunk(z_ntchw)
        return crop_spatial_padding_ntchw(rgb, pad_h, pad_w)
