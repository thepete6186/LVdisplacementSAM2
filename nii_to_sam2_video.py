"""
Convert NIfTI (.nii / .nii.gz) cines to SAM 2 video inputs.

SAM 2 `init_state(video_path=...)` accepts:
  - a folder of JPEGs named with integer stems, e.g. 0.jpg, 1.jpg, ... (sorted by int)
  - or an .mp4 file (loaded via decord inside SAM 2)

This script preserves frame order (true temporal sequence for echo cines).
Default layout is (H, W, T); set --time-axis 0 if your volume is (T, H, W).

If you want to run this script, run python nii_to_sam2_video.py --video or python nii_to_sam2_video.py --annot
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Iterator

import nibabel as nib
import numpy as np
from PIL import Image

DEFAULT_WALLMOTION_ROOT = (
    "segdata/SegmentLevelSegmentation_echo_dataset/"
    "SegmentLevelSegmentation_echo_dataset/wallMotion"
)


def _iter_time_frames(vol: np.ndarray, time_axis: int) -> Iterator[np.ndarray]:
    if vol.ndim == 2:
        yield vol
        return
    ax = time_axis if time_axis >= 0 else vol.ndim + time_axis
    if ax < 0 or ax >= vol.ndim:
        raise ValueError(f"time_axis {time_axis} invalid for shape {vol.shape}")
    for i in range(vol.shape[ax]):
        yield np.take(vol, i, axis=ax)


def _slice_to_rgb_u8(
    slice_2d: np.ndarray,
    p_low: float,
    p_high: float,
) -> np.ndarray:
    """Grayscale slice -> uint8 RGB (H, W, 3) using percentile windowing."""
    x = np.asarray(slice_2d, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        h, w = x.shape
        return np.zeros((h, w, 3), dtype=np.uint8)

    vals = x[finite]
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        u8 = np.full(x.shape, 128, dtype=np.uint8)
    else:
        y = (x - lo) / (hi - lo) * 255.0
        y = np.where(finite, y, 0.0)
        u8 = np.clip(y, 0.0, 255.0).astype(np.uint8)

    return np.stack([u8, u8, u8], axis=-1)


def export_jpeg_folder(
    frames_rgb: list[np.ndarray],
    out_dir: str,
    jpeg_quality: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = len(frames_rgb)
    width = max(1, len(str(n - 1)))
    for i, rgb in enumerate(frames_rgb):
        name = f"{i:0{width}d}.jpg"
        path = os.path.join(out_dir, name)
        Image.fromarray(rgb, mode="RGB").save(path, quality=jpeg_quality)


def export_mp4(frames_rgb: list[np.ndarray], out_path: str, fps: float) -> None:
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "MP4 output needs opencv-python. Install with: pip install opencv-python"
        ) from e

    if not frames_rgb:
        raise ValueError("no frames to write")
    h, w, _ = frames_rgb[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_path}")
    try:
        for rgb in frames_rgb:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def process_one_nii(
    path: str,
    out_root: str,
    time_axis: int,
    p_low: float,
    p_high: float,
    jpeg_quality: int,
    write_mp4: bool,
    fps: float,
) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(os.path.splitext(base)[0])[0]

    vol = nib.load(path).get_fdata(dtype=np.float32)
    frames_rgb = [
        _slice_to_rgb_u8(sl, p_low, p_high)
        for sl in _iter_time_frames(vol, time_axis)
    ]

    out_dir = os.path.join(out_root, stem)
    export_jpeg_folder(frames_rgb, out_dir, jpeg_quality=jpeg_quality)

    if write_mp4:
        mp4_path = os.path.join(out_root, f"{stem}.mp4")
        export_mp4(frames_rgb, mp4_path, fps=fps)
        print(f"{path} -> {out_dir}/ ({len(frames_rgb)} frames) + {mp4_path}")
    else:
        print(f"{path} -> {out_dir}/ ({len(frames_rgb)} frames)")

    return out_dir


def _label_slice_to_u8(slice_2d: np.ndarray) -> np.ndarray:
    x = np.asarray(slice_2d)
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, 0, 255).astype(np.uint8)


def extract_annotated_label_frames(
    label_root: str,
    out_root: str,
    time_axis: int,
) -> None:
    """Export only non-empty frames from *_label.nii.gz volumes."""
    if not os.path.isdir(label_root):
        raise FileNotFoundError(f"label root not found: {label_root}")

    os.makedirs(out_root, exist_ok=True)
    label_paths: list[str] = []
    for name in sorted(os.listdir(label_root)):
        if name.endswith("_label.nii.gz"):
            fp = os.path.join(label_root, name)
            if os.path.isfile(fp):
                label_paths.append(fp)

    if not label_paths:
        print(f"no *_label.nii.gz files found in {label_root}")
        return

    for path in label_paths:
        base = os.path.basename(path)
        stem = os.path.splitext(os.path.splitext(base)[0])[0]
        case_out = os.path.join(out_root, stem)
        os.makedirs(case_out, exist_ok=True)
        try:
            vol = nib.load(path).get_fdata(dtype=np.float32)
        except Exception as e:
            print(f"skip unreadable label volume: {base} ({e})")
            continue
        annotated_indices: list[int] = []
        for i, sl in enumerate(_iter_time_frames(vol, time_axis)):
            if not np.any(np.isfinite(sl) & (sl != 0)):
                continue
            out_path = os.path.join(case_out, f"frame_{i:03d}.png")
            Image.fromarray(_label_slice_to_u8(sl), mode="L").save(out_path)
            annotated_indices.append(i)

        print(
            f"{base} -> {case_out}/ ({len(annotated_indices)} annotated frames: "
            f"{annotated_indices})"
        )


def extract_video_frames_from_wallmotion(
    data_root: str,
    out_root: str,
    time_axis: int,
    p_low: float,
    p_high: float,
    jpeg_quality: int,
    write_mp4: bool,
    fps: float,
) -> None:
    """Export patient1..patient198 *_image.nii.gz cines from wallMotion."""
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data root not found: {data_root}")

    rows: list[tuple[int, str]] = []
    pattern = re.compile(r"^patient(\d+)-.*_image\.nii\.gz$")
    for name in sorted(os.listdir(data_root)):
        m = pattern.match(name)
        if not m:
            continue
        pid = int(m.group(1))
        if not (1 <= pid <= 198):
            continue
        fp = os.path.join(data_root, name)
        if os.path.isfile(fp):
            rows.append((pid, fp))

    if not rows:
        raise SystemExit(f"no *_image.nii.gz files found in {data_root}")

    rows.sort(key=lambda x: (x[0], os.path.basename(x[1]).lower()))
    paths = [fp for _, fp in rows]
    patient_ids = sorted({pid for pid, _ in rows})
    missing = [pid for pid in range(1, 199) if pid not in set(patient_ids)]

    os.makedirs(out_root, exist_ok=True)
    total = len(paths)
    print(
        f"found {total} image volumes in {data_root} "
        f"(patients {patient_ids[0]}..{patient_ids[-1]})"
    )
    if missing:
        print(f"warning: missing patient IDs in 1..198: {missing}")
    ok = 0
    skipped = 0
    for idx, path in enumerate(paths, start=1):
        print(f"[{idx}/{total}] processing {os.path.basename(path)}")
        try:
            process_one_nii(
                path=path,
                out_root=out_root,
                time_axis=time_axis,
                p_low=p_low,
                p_high=p_high,
                jpeg_quality=jpeg_quality,
                write_mp4=write_mp4,
                fps=fps,
            )
            ok += 1
        except Exception as e:
            skipped += 1
            print(f"skip unreadable image volume: {os.path.basename(path)} ({e})")

    print(
        "\nSAM 2: pass the per-volume JPEG folder (or .mp4) to init_state, e.g.\n"
        "  predictor.init_state(video_path=os.path.join(out_root, stem))"
    )
    print(f"completed: {ok} succeeded, {skipped} skipped")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--video",
        action="store_true",
        help="extract main cine frames from non-label .nii.gz files",
    )
    mode.add_argument(
        "--videos",
        action="store_true",
        help="alias of --video: extract all main cine frames",
    )
    mode.add_argument(
        "--annot",
        action="store_true",
        help="extract only non-empty annotated frames from *_label.nii.gz files",
    )
    p.add_argument(
        "--data-root",
        default=DEFAULT_WALLMOTION_ROOT,
        help="wallMotion directory containing both .nii.gz and _label.nii.gz files",
    )
    p.add_argument(
        "--out",
        default="Preproccessed_data/sam2_videos",
        help="output root for --video mode (one JPEG folder per volume)",
    )
    p.add_argument(
        "--time-axis",
        type=int,
        default=-1,
        help="axis to treat as time (default -1 = last axis, typical for HxWxT)",
    )
    p.add_argument(
        "--p-low",
        type=float,
        default=1.0,
        help="low percentile for intensity window (default 1)",
    )
    p.add_argument(
        "--p-high",
        type=float,
        default=99.0,
        help="high percentile for intensity window (default 99)",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality 1-95 (default 95)",
    )
    p.add_argument(
        "--mp4",
        action="store_true",
        help="also write <out>/<stem>.mp4 (requires opencv-python)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="frames per second for --mp4 only (default 30)",
    )
    p.add_argument(
        "--annotated-out",
        default="Preproccessed_Data/Annotated_frames",
        help="output root for --annot mode",
    )
    args = p.parse_args()

    if args.annot:
        extract_annotated_label_frames(
            label_root=args.data_root,
            out_root=args.annotated_out,
            time_axis=args.time_axis,
        )
        return

    extract_video_frames_from_wallmotion(
        data_root=args.data_root,
        out_root=args.out,
        time_axis=args.time_axis,
        p_low=args.p_low,
        p_high=args.p_high,
        jpeg_quality=args.jpeg_quality,
        write_mp4=args.mp4,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
