"""Resume support for tiled runs: manifest, temp tile directory, completed tiles.

A tiled run writes one temporary mp4 per tile into ``tiles_<hash>``, where the
hash covers every setting that affects tile pixels. A tile counts as complete
only when its ``.done`` marker exists and records the frame count found on disk.
"""

import os
import json
import shutil
import hashlib
import datetime
from pathlib import Path

import decord

TEMP_DIR_NAME = "_swiftvr_temp"
TILE_DIR_PREFIX = "tiles_"


def input_fingerprint(input_path, image_frames=None) -> dict:
    """Identify the input. ``st_mtime_ns`` (int) instead of ``st_mtime``: floats
    do not survive a JSON round-trip bit-exactly."""
    path = Path(input_path)
    if image_frames is not None:
        return {"type": "images", "path": str(path.resolve()),
                "files": [[p.name, p.stat().st_size] for p in image_frames]}
    st = path.stat()
    return {"type": "video", "path": str(path.resolve()),
            "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def build_resume_manifest(params: dict, info: dict) -> dict:
    """``params`` holds everything that affects tile pixels, coordinates or
    count; two runs with equal params produce interchangeable tile videos.
    Stitch-only settings (output path, quality, ``output_height``, fps) stay out
    of it so changing them does not discard completed tiles. ``info`` is for
    human inspection only."""
    info = dict(info, created_at=datetime.datetime.now().isoformat(timespec="seconds"))
    return {"format": 1, "params": params, "info": info}


def resume_manifest_hash(manifest: dict) -> str:
    blob = json.dumps(manifest["params"], sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def tile_video_path(tile_dir, index: int) -> str:
    return os.path.join(str(tile_dir), f"{index + 1:05d}.mp4")


def prepare_resume_dir(tile_dir, manifest: dict, resume: bool) -> bool:
    """Ensure ``tile_dir`` exists. Returns True when ``resume`` is set and the
    manifest on disk has exactly the same params (completed tiles may be
    reused); otherwise the directory is recreated with a fresh manifest. Only
    this run's own ``tiles_<hash>`` directory is ever removed."""
    tile_dir = str(tile_dir)
    manifest_path = os.path.join(tile_dir, "manifest.json")
    if resume and os.path.isdir(tile_dir):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                on_disk = json.load(f)
        except (OSError, ValueError):
            on_disk = None
        if isinstance(on_disk, dict) and on_disk.get("params") == manifest["params"]:
            return True
        print("[swiftvr] Warning: --resume: the previous tiles were made with different "
              "parameters; discarding them.", flush=True)
    if os.path.exists(tile_dir):
        shutil.rmtree(tile_dir)
    os.makedirs(tile_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return False


def count_video_frames(path):
    """Frame count via decord, or None if the file cannot be opened (e.g. a
    killed process left an mp4 without its moov atom)."""
    try:
        vr = decord.VideoReader(str(path))
        n = len(vr)
    except Exception:
        return None
    # decord keeps the file open until the reader is freed; Windows cannot
    # delete an open file.
    del vr
    return n


def write_done_marker(tile_dir, index: int, frames: int) -> None:
    with open(tile_video_path(tile_dir, index) + ".done", "w", encoding="utf-8") as f:
        json.dump({"frames": int(frames)}, f)


def scan_completed_tiles(tile_dir, num_tiles: int, total_frames: int, verbose: bool = True) -> set:
    """Indices of tiles whose mp4 is complete: the ``.done`` marker exists, its
    frame count equals ``total_frames`` and the frame count on disk. The runner
    closes its writer in a ``finally`` block, so a crashed tile leaves a
    readable but short mp4. Everything that fails the checks is deleted."""
    completed = set()
    for i in range(num_tiles):
        mp4 = tile_video_path(tile_dir, i)
        marker = mp4 + ".done"
        if os.path.exists(mp4) and os.path.exists(marker):
            try:
                with open(marker, encoding="utf-8") as f:
                    recorded = json.load(f).get("frames")
            except (OSError, ValueError, AttributeError):
                recorded = None
            n = count_video_frames(mp4)
            if isinstance(recorded, int) and recorded == total_frames and n == recorded:
                completed.add(i)
                continue
            if verbose:
                print(f"[swiftvr] Tile {i + 1}: {n} frames on disk, marker records {recorded}, "
                      f"expected {total_frames}; recomputing it.", flush=True)
        for path in (mp4, marker):
            if os.path.exists(path):
                os.remove(path)
    return completed


def list_stale_tile_dirs(temp_root, keep) -> list:
    """Other ``tiles_*`` directories under ``temp_root`` (left by earlier runs or
    running concurrently); they are reported, never removed."""
    root = Path(temp_root)
    if not root.is_dir():
        return []
    keep = Path(keep).resolve()
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and p.name.startswith(TILE_DIR_PREFIX) and p.resolve() != keep)
