<h1 align="center">SwiftVR: Real-Time One-Step Generative Video Restoration</h1>

<p align="center"><img src="assets/teaser.avif" width="100%" alt="SwiftVR teaser"></p>

> **SwiftVR** is the first generative video restoration model to reach **real-time 1080p streaming on a consumer-grade GPU** (≈26 FPS on a single RTX 5090), sustains **31 FPS at QHD (2560×1440)** and **14 FPS at 4K (3840×2160)** on a single H100, and streams at resolutions where every compared diffusion-based VR baseline runs out of memory.

<p>
  <a href="https://arxiv.org/abs/2606.09516"><img src="https://img.shields.io/badge/arXiv-2606.09516-b31b1b.svg?style=flat-square" alt="arXiv"></a>
  <a href="https://h-oliday.github.io/SwiftVR"><img src="https://img.shields.io/badge/Project-Page-1f8acb.svg?style=flat-square" alt="Project Page"></a>
  <a href="https://huggingface.co/H-oliday/SwiftVR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-ffce00.svg?style=flat-square" alt="HuggingFace"></a>
  <a href="https://github.com/H-oliday/SwiftVR/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" alt="License"></a>
</p>

## About this fork (`modi` branch)

This is a fork of [H-oliday/SwiftVR](https://github.com/H-oliday/SwiftVR).
`main` mirrors upstream as-is; all changes live on `modi` (the default branch).
The model and checkpoint are unchanged; only packaging and inference code differ.

* **uv packaging and CLI.** `pyproject.toml` + `uv.lock` replace `setup.py`
  (Python 3.12+, torch from the CUDA 13.2 index for Blackwell / RTX 50xx), and a
  `swiftvr` command replaces `python scripts/inference.py` (kept as a wrapper).
  Runs on Windows without a C++ compiler.
* **Lower GPU memory (device-independent).** The autoencoder processes its
  stateless layers a few frames at a time, runs the decoder's full-resolution tail
  only on the frames that are kept, and applies `TGrow` before the nearest
  upsample (at 1/4 the pixels). The unfused Q/K/V weights are freed after QKV
  fusion (~2.6 GiB). The first two come from
  [LightX2V](https://github.com/ModelTC/LightX2V).
* **Faster inference (CUDA only; other devices fall back to the original path).**
  * cuDNN attention picked automatically when FlashAttention/SageAttention is absent
  * fused channels-last cuDNN convolutions in the autoencoder (`--no-reae-fusion` to disable)
  * opt-in FP8 DiT (`--fp8-dit`, RTX 40 series or newer)
  * INT8 attention via comfy-kitchen (`uv sync --extra kitchen`; picked by `auto` once installed)
  * `--torch_compile` working on Windows via `triton-windows`

  On an RTX 5060 Ti 16GB (640×480 → 1280×960) throughput goes from 9.1 to
  23.3 fps and peak memory from 12.0 to 8.0 GiB; see
  [Command line](#command-line) for the per-option table.
* **`cudnn.benchmark` off by default** (`--cudnn-benchmark` to enable). This makes
  the output bit-identical across runs and avoids several GiB of peak memory from
  the algorithm search.
* **Spatial tiling** (`--tiled-dit`) for output resolutions that do not fit in GPU
  memory, with resumable runs; ported from FlashVSR_plus. See
  [Tiled processing](#tiled-processing).
* **`--pad-align`** keeps the whole frame (the input is otherwise cropped to a
  multiple of 8) and pads with reflection instead of black.

Output is not bit-identical to upstream: the trained DiT amplifies small bf16
differences, and FP8 differs from bf16 by about 47 dB PSNR. In a side-by-side
visual comparison, no difference was visible between the configurations above.

Tested on Windows 11 with an RTX 5060 Ti 16GB (Python 3.13) and on Ubuntu 26.04
with an RTX 5080 16GB (Python 3.14). The measurements in this README come from
the Windows machine. Apple Silicon (MPS) has not been run.

## Updates

* [2026/06] Release the inference code and pretrained weights 🎉

## Community Works

* [LightX2V](https://github.com/ModelTC/LightX2V) brings **faster inference and lower GPU memory usage** to SwiftVR (**1.91× speedup and 65.71% lower peak GPU memory per request** on a single H100). It also supports **multi-GPU acceleration**. Ready-to-use scripts cover image and video super-resolution, offline inference, and API serving. **[Get started with LightX2V →](https://github.com/ModelTC/LightX2V/tree/main/scripts/swiftvr)**

## ✨ Highlights

* **Mask-free shifted-window self-attention (MFSWA).** Each spatial window is **pre-gathered into a dense tensor**, so every attention call reduces to a single standard scaled-dot-product (SDPA) call — *no attention mask, cyclic shift, or padding ever enters the graph*. This gives a **1.62× throughput gain over its full-attention teacher** at essentially identical quality, with **no dedicated sparse kernel**.
* **Restoration-aware Autoencoder (ReAE).** A lightweight encoder–decoder jointly fine-tuned with the DiT in pixel space removes the heavy-3D-VAE / tiled-decoding bottleneck.
* **Causal chunk-wise streaming.** A minimal causal protocol (no rolling KV cache, no overlapped DiT inference) bounds the temporal axis, confining the residual (\mathcal{O}(N^2)) cost to the spatial axes.


## 📊 Results

### Efficiency at 2560×1440

Single H100, causal streaming, 24 frames.

| Metric | DOVE (tile) | SeedVR2-3B (tile)| FlashVSR-Tiny | **SwiftVR (Ours)** |
|---|:---:|:---:|:---:|:---:|
| Avg. Time (s) ↓ | 27.615 | 17.320 | 2.493 | 0.766 |
| FPS ↑ | 0.85 | 1.39 | 9.61 | 31.32 |
| Peak Mem. (GB) ↓ | 59.24 | 35.35 | 34.35 | 38.01 |

> At **3840×2160**, every compared diffusion-based VR baseline **OOMs** on a single H100; SwiftVR sustains **14 FPS**.

### Qualitative comparison

<img src="assets/qualitative.png" width="100%" alt="SwiftVR qualitative comparison">

## 🛠 Installation

```bash
git clone https://github.com/sh202603/SwiftVR.git   # default branch: modi
cd SwiftVR

# With uv (Windows / Linux). torch is pulled from the CUDA 13.2 index configured
# in pyproject.toml (required for Blackwell GPUs such as RTX 50xx).
uv sync                       # creates .venv (Python >= 3.12) and installs `swiftvr`
uv sync --extra kitchen       # optional: comfy-kitchen INT8 attention (--attention_backend kitchen)
uv run swiftvr --help

# Or install the `swiftvr` command globally on PATH:
uv tool install -e .
```

With pip instead: install torch from the CUDA index matching your driver
(e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu132`), then `pip install -e .`.

<details>
<summary><b>Hardware notes</b></summary>

* **Server:** single H100-80G reproduces the QHD/4K numbers above.
* **Consumer:** single RTX 5090 reaches ≈26 FPS at 1080p with the *same checkpoint* (default PyTorch SDPA path, bfloat16, causal chunk protocol).
* No hardware-specific retraining or kernel rewrite is required on any platform.

</details>

## 🗂 Model Zoo

| Model Name | Date    | Backbone       | Link                                                  |
| ---------- | ------- | -------------- | ----------------------------------------------------- |
| SwiftVR        | 2026.06 | Wan2.2-TI2V-5B | [🤗 HuggingFace](https://huggingface.co/H-oliday/SwiftVR) |

```bash
uv run hf download H-oliday/SwiftVR --local-dir checkpoints/
```

Expected checkpoint layout, where `checkpoints/` is the directory passed to `from_pretrained`:

```text
checkpoints/
├── reae.safetensors             # Restoration-aware Autoencoder weights
├── prompt_embedding.safetensors # precomputed empty-prompt text embedding, key: "prompt_emb"
└── transformer/                 # diffusers-format DiT
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

## 🚀 Quick Start

### Python API

```python
from swiftvr import SwiftVRPipeline

pipe = SwiftVRPipeline.from_pretrained("checkpoints/").to("cuda", dtype="bfloat16")

pipe.restore_video("low_quality.mp4", "restored.mp4", upscale=4)
```

`restore_video` also accepts an image folder as input and can write a PNG sequence with `png_save=True`.

Tunable knobs include:

* `clip_len`: middle chunk size, multiple of 4
* `dit_overlap`: overlap for DiT inference
* `fps`: output video frame rate
* `quality`: 0–100, mapped to x265 CRF
* `queue_size`: pipeline queue size
* `copy_audio`: copy the input video's audio tracks into the output mp4 (default on)

### One clip in memory

`restore_clip` restores a clip that is already in memory and returns exactly
as many frames as it was given, for any length from one frame up. It exists
for callers that upscale many short clips (such as
[jasna](https://github.com/sh202603/jasna)'s secondary restoration, which
hands over the mosaic crops of one detection track at a time).

```python
lq = ...                                            # [T, H, W, 3] uint8, CPU or CUDA
hq = pipe.restore_clip(lq, upscale=4)               # [T, 4H, 4W, 3] uint8 on pipe.device
```

The clip goes through the same fixed chunk protocol and code as
`restore_video`, so a `4k+1`-frame clip gives the same frames as the file path
does. Shorter or other lengths are padded by repeating the last frame: up to
the protocol's `4k+1` and to at least `clip_len + 1` frames (the `min_frames`
argument). The floor matters for short clips: anything up to `clip_len + 4`
frames is a single LAST chunk whose DiT input always holds `clip_len / 4 + 1`
latents, and without the floor the missing latents would be zeros, while with
it they come from (repeated) real frames at the same DiT cost. The input is
not cropped to a multiple of 8; temporal overlap is not used. `clip_len` is
the same knob as for `restore_video`.

### Streaming

Causal, chunk-by-chunk restoration without future frames.

```python
session = pipe.stream(clip_len=24, resolution=(1920, 1080))

for lq_chunk in read_chunks("low_quality.mp4", n=24):   # lq_chunk: [T, H, W, 3] uint8
    hq = session.step(lq_chunk)                         # [1, T', 3, H', W'] in [0, 1], or None if buffered
    if hq is not None:
        write(hq)

tail = session.flush()                                  # flush the final buffered frames
```

### Command line

```bash
swiftvr \
  --input low_quality.mp4 \
  --output restored.mp4 \
  --checkpoint checkpoints/ \
  --upscale 4 \
  --clip-len 24 \
  --dtype bfloat16
```

Use `--png` to write a PNG sequence. `python scripts/inference.py` accepts the same arguments.

The audio tracks of a video input are copied into the output mp4 after restoration
(stream copy, no re-encoding; `--no-audio` turns this off). If a track cannot be
stored in the output container the output is kept without audio and a warning is
printed. The video is up to 3 frames shorter than the input (frame count truncated
to `4k+1`), so the audio may outlast the picture by that much.

Memory knobs: `--reae-frame-batch-size` (default 2) bounds how many frames the
autoencoder's stateless layers process at once; `--cudnn-benchmark` is off by
default because its conv-algorithm search adds several GiB of peak memory.

Speed knobs (RTX 5060 Ti 16GB, 640×480 → 1280×960, steady-state GPU throughput):

| Options | GPU fps | Peak memory |
| --- | --- | --- |
| `--attention_backend sdpa --no-reae-fusion` (previous defaults) | 9.1 | 12.0 GiB |
| `--fp8-dit --no-reae-fusion` | 15.5 | 7.4 GiB |
| `--fp8-dit` | 17.2 | 8.0 GiB |
| `--fp8-dit --torch_compile` | 23.3 | 8.0 GiB |

* The default attention backend `auto` picks PyTorch's cuDNN attention when
  FlashAttention/SageAttention is not installed.
* The autoencoder runs channels-last with fused cuDNN convolutions by default
  (encoder 110 → 60 ms, decoder 346 → 244 ms per 24-frame chunk, ~0.6 GiB more
  memory). `--no-reae-fusion` restores the plain PyTorch path.
* `--fp8-dit` runs the DiT's linear layers in FP8 (RTX 40 series or newer, bfloat16).
  Its output differs slightly from bfloat16 (≈47 dB PSNR against the bfloat16 output).
* The `kitchen` backend runs attention with INT8-quantized Q/K via
  [comfy-kitchen](https://github.com/comfy-org/comfy-kitchen) (`uv sync --extra kitchen`;
  prebuilt CUDA wheels for Windows and Linux). On an RTX 5080 16GB
  (640×480 → 1280×960) it raised eager throughput from 12.0 to 17.0 fps in
  bfloat16 and from 19.6 to 25.7 fps with `--fp8-dit`; combined with
  `--torch_compile` the gain shrinks to a few percent. Output stays bit-identical
  across runs; the difference against the cuDNN backend is at the model's noise
  floor (≈53 dB PSNR), and a visual A/B found no difference. Installing the
  extra is the opt-in: once present, `auto` prefers it over every other backend.
  Pass `--attention_backend cudnn` to reproduce the exact pre-kitchen output
  (note that tiled `--resume` runs started without comfy-kitchen need this flag
  to reuse their tiles, because the resolved backend is part of the tile hash).
* `--torch_compile` fuses the DiT's elementwise ops. Compiling takes the first two
  chunks (roughly 10–20 s), so it pays off only on longer videos. On Windows it uses
  `triton-windows` (installed by `uv sync`); no C++ compiler is needed. On Linux,
  Triton builds a small C launcher on first use and needs the Python headers
  (e.g. `python3.14-dev` for a venv on the system Python 3.14); this also applies
  to `--fp8-dit`.

Frame edges: the input is cropped to a multiple of 8 pixels (e.g. width 1918 →
1912), and the upscaled frame is zero-padded on the right and bottom to a multiple
of 32 before processing. `--pad-align` (`pad_align=True`) crops only to even sizes
and pads by reflection instead. It changes the output, so it is off by default.

### Tiled processing

Peak GPU memory is set by the output resolution. For outputs that do not fit,
`--tiled-dit` splits the input into overlapping square tiles, restores each tile
over the whole video into a temporary mp4, and blends the tiles with feather
weights into the final output.

```bash
swiftvr --input 1080p.mp4 --output 4320p.mp4 --checkpoint checkpoints/ \
  --upscale 4 --fp8-dit --tiled-dit --tile-size 256 --overlap 24
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--tiled-dit` | off | Enable tiling (all stages, not only the DiT; the name matches FlashVSR_plus). |
| `--tile-size` | 256 | Tile size in input pixels. `tile-size × upscale` must be a multiple of 32. |
| `--overlap` | 24 | Overlap between tiles in input pixels (at most half the tile size). |
| `--output-height` | native | Downscale the blended result to this height (area filter). |
| `--temp-quality` | 80 | Quality 0–100 of the temporary tile videos (x265, yuv444p). |
| `--resume` | off | Reuse the tiles that an interrupted run with the same settings completed. If all tiles are complete, the model is not loaded. |
| `--temp-dir` | see below | Where the temporary tile videos go. |

The Python API takes the same settings: `restore_video(..., tile_size=256,
tile_overlap=24, output_height=None, temp_quality=80, resume=False, temp_dir=None)`.

* Keep `tile-size × upscale` at 1024 or more. The shifted-window attention cannot
  shift inside smaller tiles (a warning is printed). With `--upscale 4` this means
  `--tile-size 256` or more.
* Per-tile peak GPU memory (RTX 5060 Ti, measured as allocated memory):

  | Tile output | FP8 DiT | bfloat16 |
  | --- | --- | --- |
  | 1024×1024 (`--tile-size 256`, 4×) | 7.5 GiB | 12.1 GiB |
  | 1280×1280 (`--tile-size 320`, 4×) | 9.0 GiB | 13.6 GiB |
  | 1536×1536 (`--tile-size 384`, 4×) | 10.8 GiB | out of memory on 16 GB |

* `--resolution` cannot be combined with tiling, because the upscale factor must be
  an integer. Use `--output-height` to change the output size.
* The input is decoded once per tile. A 1080p input at `--upscale 4` with the
  default tile size is 45 tiles, about 1.4× the pixels of an untiled run.
* Temporary videos go to `_swiftvr_temp/tiles_<hash>` next to the output mp4 (for
  `--png`, next to the PNG folder), or under `--temp-dir`. The hash covers every
  setting that affects the tile pixels. Only this run's directory is deleted, after
  stitching; directories left by other runs are listed in the log but not removed.
* The temporary videos are lossy even at `--temp-quality 100` (x265 CRF 0 is not
  lossless), including for `--png` output. At quality 80 they took about
  0.65 MiB per second per output megapixel on a test clip.
* Where the tiles overlap, the result is a weighted average of two tiles. Averaging
  two slightly different generative outputs may soften detail there.

## 📁 Repository Structure

```text
SwiftVR/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml                # package metadata, `swiftvr` console script, uv torch index
├── scripts/
│   └── inference.py              # backward-compatible wrapper over swiftvr.cli
└── swiftvr/
    ├── __init__.py               # exports SwiftVRPipeline
    ├── cli.py                    # `swiftvr` command-line entry point
    ├── pipeline.py               # SwiftVRPipeline: from_pretrained / to / restore_video / stream
    ├── runner.py                 # four-stage pipelined runner: reader → H2D → GPU → writer
    ├── io.py                     # frame reading, GPU preprocessing, mp4 / PNG writing
    ├── tiling.py                 # spatial tiling: tile layout, feather blending, stitching
    ├── resume.py                 # resumable tiled runs: manifest, temp tile directory
    ├── models/
    │   ├── fp8.py                # opt-in FP8 linear / feed-forward layers for the DiT
    │   ├── reae.py               # Restoration-aware Autoencoder
    │   └── transformer.py        # DiT + mask-free shifted-window self-attention
    └── streaming/
        ├── chunk.py              # fixed-size causal chunk protocol
        ├── tae.py                # streaming autoencoder with causal boundary state
        └── dit.py                # one-step streaming DiT with fixed timestep and RoPE offsets
```

## 🙏 Acknowledgements

SwiftVR builds on [Wan2.2-TI2V-5B](https://github.com/Wan-Video), the lightweight autoencoder [TAEHV](https://github.com/madebyollin/taehv), and the [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) degradation pipeline.

We thank the authors of [DOVE](https://github.com/zhengchen1999/DOVE), [SeedVR2](https://github.com/ByteDance-Seed/SeedVR), and [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) for releasing strong baselines, and the [UltraVideo](https://huggingface.co/datasets/APRIL-AIGC/UltraVideo) team for the training corpus.

## 📜 License

SwiftVR is released under the **Apache License 2.0**.

Copyright 2026 SwiftVR Authors.

Licensed under the Apache License, Version 2.0. You may obtain a copy of the License at:

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, this project is distributed on an **"AS IS" BASIS**, without warranties or conditions of any kind, either express or implied. See the [LICENSE](./LICENSE) file for the full license text.



## Contact

If you have any questions, feel free to reach out:

* Email: [kakibluee@gmail.com](mailto:kakibluee@gmail.com)

<div align="center">
<sub>If SwiftVR is useful to your research or product, please consider giving it a ⭐.</sub>
</div>
