"""Command-line entry point for SwiftVR inference (``swiftvr`` console script).

Thin wrapper around ``swiftvr.SwiftVRPipeline``.

    swiftvr --input low_quality.mp4 --output restored.mp4 \
        --checkpoint checkpoints/ --upscale 4 --clip-len 24 --dtype bfloat16
"""

import argparse

from .pipeline import SwiftVRPipeline, DEFAULT_REAE_FRAME_BATCH_SIZE


def _parse_resolution(value):
    if value is None:
        return None
    w, h = value.lower().split("x")
    return int(w), int(h)


def build_parser():
    p = argparse.ArgumentParser(description="SwiftVR streaming video restoration.")

    p.add_argument("--input", required=True, help="Low-quality video file or image folder.")
    p.add_argument("--output", required=True, help="Output mp4 path or directory.")
    p.add_argument("--checkpoint", required=True, help="Checkpoint directory (see README layout).")

    p.add_argument("--resolution", type=str, default=None,
                   help="Output resolution as WxH (e.g. 1920x1080). Overrides --upscale.")
    p.add_argument("--upscale", type=int, default=4, help="Upscale factor when --resolution is unset.")
    p.add_argument("--clip-len", type=int, default=24, help="MIDDLE chunk size (multiple of 4).")
    p.add_argument("--dit-overlap", type=int, default=0, help="Temporal overlap (latents) for blending.")

    p.add_argument("--fps", type=float, default=None, help="Output fps (defaults to source fps).")
    p.add_argument("--quality", type=int, default=60, help="Output quality 0-100 (maps to x265 CRF).")
    p.add_argument("--png", action="store_true", help="Write a PNG sequence instead of an mp4.")
    p.add_argument("--save-format", type=str, default="", help="Set to 'yuv444p' for 4:4:4 mp4.")
    p.add_argument("--ffmpeg-preset", type=str, default="", help="x265 preset (e.g. fast, medium).")
    p.add_argument("--queue-size", type=int, default=3, help="Pipeline queue depth.")
    p.add_argument("--reae-frame-batch-size", type=int, default=DEFAULT_REAE_FRAME_BATCH_SIZE,
                   help="Frames per batch in the autoencoder's stateless layers; lower saves "
                        "GPU memory. 0 processes a whole chunk at once.")
    p.add_argument("--no-reae-fusion", action="store_true",
                   help="Run the autoencoder with plain PyTorch ops (NCHW) instead of channels-last "
                        "fused cuDNN convolutions.")
    p.add_argument("--attention_backend", type=str, default="auto", choices=["auto", "sdpa", "cudnn", "flash_attn_2", "flash_attn_3", "sageattention", "xformers"],
                    help="Attention backend. 'auto' lets SwiftVR pick the fastest available backend.",)
    p.add_argument("--torch_compile", action="store_true", help="Enable torch.compile. Disabled by default to avoid long recompilation on dynamic paths.",)
    p.add_argument("--fp8-dit", action="store_true",
                   help="Run the DiT's linear layers in FP8 (RTX 40 series or newer, bfloat16): "
                        "faster and ~4.6 GiB less GPU memory, slightly different output.")
    p.add_argument("--cudnn-benchmark", action="store_true",
                   help="Enable cudnn.benchmark (may be faster on long videos, costs several GiB of peak memory).")
    p.add_argument("--pad-align", action="store_true",
                   help="Keep the whole frame: crop the input to even sizes instead of multiples "
                        "of 8, and reflect-pad (instead of zero-pad) the upscaled frame to a "
                        "multiple of 32.")

    t = p.add_argument_group("tiling (high output resolutions)")
    t.add_argument("--tiled-dit", action="store_true",
                   help="Restore the video in overlapping spatial tiles (all stages, not only "
                        "the DiT) and blend them. Peak memory then follows the tile size.")
    t.add_argument("--tile-size", type=int, default=256,
                   help="Tile size in input pixels; tile-size x upscale must be a multiple of 32.")
    t.add_argument("--overlap", type=int, default=24, help="Tile overlap in input pixels.")
    t.add_argument("--output-height", type=int, default=None,
                   help="Downscale the stitched output to this height (area filter).")
    t.add_argument("--temp-quality", type=int, default=80,
                   help="Quality 0-100 of the temporary per-tile mp4s (yuv444p x265; not lossless).")
    t.add_argument("--resume", action="store_true",
                   help="Reuse the tiles an interrupted run with identical settings completed.")
    t.add_argument("--temp-dir", type=str, default=None,
                   help="Directory for the temporary tile videos (default: _swiftvr_temp next to "
                        "the output mp4, or next to the PNG folder).")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--quiet", action="store_true")
    return p


def _print_stats(stats):
    print(f"\nDone. {stats['frames']} frames in {stats['seconds']:.2f}s "
          f"({stats['fps']:.2f} fps) -> {stats['output']}")
    if stats["max_memory_reserved"]:
        gib = 1024 ** 3
        print(f"Peak GPU memory: allocated {stats['max_memory_allocated'] / gib:.2f} GiB, "
              f"reserved {stats['max_memory_reserved'] / gib:.2f} GiB")


def _stitch_only_if_complete(args) -> bool:
    """Plan the tiled run before loading the model: invalid tiling settings fail
    fast, and a resumed run whose tiles are all complete is stitched without
    loading the model. Returns True if it stitched."""
    import time
    from .pipeline import _as_dtype
    from .tiling import plan_tiled_run, finish_tiled_run, _log

    t_start = time.perf_counter()
    plan = plan_tiled_run(
        args.input, args.output, tile_size=args.tile_size, tile_overlap=args.overlap,
        upscale=args.upscale, resolution=_parse_resolution(args.resolution),
        clip_len=args.clip_len, dit_overlap=args.dit_overlap, fps=args.fps, png_save=args.png,
        pad_align=args.pad_align, dtype=str(_as_dtype(args.dtype)),
        attention_backend=args.attention_backend, fp8_dit=args.fp8_dit,
        torch_compile=args.torch_compile, reae_fused=not args.no_reae_fusion,
        reae_frame_batch_size=args.reae_frame_batch_size or None,
        cudnn_benchmark=args.cudnn_benchmark, temp_quality=args.temp_quality, resume=args.resume,
        temp_dir=args.temp_dir, verbose=not args.quiet)
    if not (args.resume and plan.all_complete):
        return False
    _log("All tiles already complete; skipping the model load and stitching.", not args.quiet)
    stats = finish_tiled_run(plan, quality=args.quality, save_format=args.save_format,
                             ffmpeg_preset=args.ffmpeg_preset, output_height=args.output_height,
                             device=args.device, verbose=not args.quiet, t_start=t_start)
    _print_stats(stats)
    return True


def main():
    args = build_parser().parse_args()

    if args.tiled_dit and _stitch_only_if_complete(args):
        return

    pipe = SwiftVRPipeline.from_pretrained(
        args.checkpoint, reae_frame_batch_size=args.reae_frame_batch_size or None,
        reae_fused=not args.no_reae_fusion).to(
        args.device, dtype=args.dtype,
        attention_backend=args.attention_backend,
        torch_compile=args.torch_compile,
        cudnn_benchmark=args.cudnn_benchmark,
        fp8_dit=args.fp8_dit)

    stats = pipe.restore_video(
        args.input, args.output,
        resolution=_parse_resolution(args.resolution),
        upscale=args.upscale,
        clip_len=args.clip_len,
        dit_overlap=args.dit_overlap,
        fps=args.fps,
        quality=args.quality,
        png_save=args.png,
        save_format=args.save_format,
        ffmpeg_preset=args.ffmpeg_preset,
        queue_size=args.queue_size,
        verbose=not args.quiet,
        pad_align=args.pad_align,
        tile_size=args.tile_size if args.tiled_dit else None,
        tile_overlap=args.overlap,
        output_height=args.output_height,
        temp_quality=args.temp_quality,
        resume=args.resume,
        temp_dir=args.temp_dir,
    )
    _print_stats(stats)


if __name__ == "__main__":
    main()
