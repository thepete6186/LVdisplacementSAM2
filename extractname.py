"""
Dataset helpers for LVdisplacement.

This script intentionally has **no positional arguments**. Use one of:

  - ``python extractname.py --inventory``
      Prints an inventory of the extracted SAM2 video frame folders under
      ``Preproccessed_data\\sam_2_videos`` (or ``sam2_videos``).

  - ``python extractname.py --annot``
      For each ``*_label.nii.gz`` inside the wallMotion dataset folder, extracts
      **two** annotated frames (the first two non-empty time frames) and writes:

        Preprocessed_Data\\Annotated-frames\\<stem>\\frame_XXX.png

Historical behavior (applying annotated PNGs to roughmasks) is still available
via ``--apply-roughmasks``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Repo root (folder that contains this file and Preprocessed_Data/)
_PROJECT = Path(__file__).resolve().parent
DEFAULT_SAM2_VIDEOS = _PROJECT / "Preproccessed_data" / "sam2_videos"
DEFAULT_PREPROCCESSED_DATA = _PROJECT / "Preproccessed_Data"
DEFAULT_PREPROCCESSED_DATA = _PROJECT / "Preproccessed_Data"
DEFAULT_WALLMOTION_ROOT = (
    _PROJECT
    / "segdata"
    / "SegmentLevelSegmentation_echo_dataset"
    / "SegmentLevelSegmentation_echo_dataset"
    / "wallMotion"
)


@dataclass(frozen=True)
class VideoEntry:
    """One discovered frame sequence (SAM2: directory of numbered JPEGs)."""

    root: Path
    """Absolute path to the folder containing frame JPEGs."""

    indicator: str
    """Leading token: folder name minus trailing ``_image`` when convention matches."""

    folder_name: str
    """Basename of ``root`` (e.g. ``patient96-MCE-A3C_image``)."""

    matched_indicator_convention: bool
    """True if ``folder_name`` endswith ``_image``."""

    frame_count: int
    """Number of .jpg/.jpeg files in this folder (non-recursive)."""


def _has_frame_jpegs(d: Path) -> bool:
    if not d.is_dir():
        return False
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"):
            return True
    return False


def parse_indicator_from_folder_name(folder_name: str) -> tuple[str, bool]:
    """
    ``patient96-MCE-A3C_image`` -> (``patient96-MCE-A3C``, True).
    ``other`` -> (``other``, False).
    """
    suffix = "_image"
    if folder_name.endswith(suffix):
        return folder_name[: -len(suffix)], True
    return folder_name, False


def discover_sam2_videos(video_root: Path) -> list[VideoEntry]:
    """
    Walk the entirety of ``video_root`` and return every directory that directly
    contains at least one JPEG frame (same rule as SAM2 JPEG folder loading).
    """
    if not video_root.is_dir():
        return []

    frame_dirs: set[Path] = set()
    for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        for f in video_root.rglob(pattern):
            if f.is_file():
                frame_dirs.add(f.resolve().parent)

    entries: list[VideoEntry] = []
    for d in sorted(frame_dirs, key=lambda p: str(p).lower()):
        if not _has_frame_jpegs(d):
            continue
        name = d.name
        indicator, ok = parse_indicator_from_folder_name(name)
        n_frames = sum(
            1
            for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
        )
        entries.append(
            VideoEntry(
                root=d,
                indicator=indicator,
                folder_name=name,
                matched_indicator_convention=ok,
                frame_count=n_frames,
            )
        )
    return entries

def print_video_inventory(entries: list[VideoEntry], video_root: Path) -> None:
    print()
    print(f"SAM2 video root: {video_root}")
    print(f"Total videos (frame folders): {len(entries)}")
    print("-" * 72)
    for e in entries:
        rel = e.root.relative_to(video_root) if e.root.is_relative_to(video_root) else e.root
        flag = "OK" if e.matched_indicator_convention else "name?"
        print(
            f"  [{e.indicator!r}]  {e.folder_name!r}  "
            f"({e.frame_count} frames)  [{flag}]  -> {rel}"
        )
    print("-" * 72)


def _strip_suffixes(name: str) -> str:
    # Normalize names across different folders.
    for suf in ("_image", "_label", "_prediction", "_roughmask"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name


def _parse_annot_frame_index(filename: str) -> int | None:
    # Expected: frame_037.png
    m = re.match(r"^frame_(\d+)\.png$", filename, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def _find_roughmask_target(rough_dir: Path, frame_idx: int) -> Path:
    """
    Roughmask frame filename can vary; try common conventions.
    Preference order: <idx>.png (SAM2-style), then frame_<idx>.png.
    """
    candidates = [
        rough_dir / f"{frame_idx:03d}.png",
        rough_dir / f"{frame_idx}.png",
        rough_dir / f"frame_{frame_idx:03d}.png",
        rough_dir / f"frame_{frame_idx}.png",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return rough_dir / f"{frame_idx:03d}.png"


def apply_annotations_to_roughmasks(
    rough_root: Path | None = None,
    annotated_root: Path | None = None,
) -> None:
    """
    For each indicator, if an annotated frame exists (frame_XXX.png), replace the
    corresponding roughmask frame with that PNG (renamed to match roughmask naming).

    Expected layout:
      Preproccessed_Data/sam2roughmasks/<indicator>_roughmask/*.png
      Preproccessed_Data/Annotated_frames/<matching indicator folder>/frame_XXX.png
    """
    rough_root = (rough_root or (DEFAULT_PREPROCCESSED_DATA / "sam2roughmasks")).resolve()
    annotated_root = (annotated_root or (DEFAULT_PREPROCCESSED_DATA / "Annotated_frames")).resolve()

    if not rough_root.is_dir():
        raise FileNotFoundError(f"missing roughmask root: {rough_root}")
    if not annotated_root.is_dir():
        raise FileNotFoundError(f"missing annotated root: {annotated_root}")

    rough_dirs = sorted(
        [
            p
            for p in rough_root.iterdir()
            if p.is_dir() and (p.name.endswith("_roughmask") or p.name.endswith("_roughmasks"))
        ]
    )
    if not rough_dirs:
        print(f"no roughmask folders found in {rough_root}")
        return

    # Case-insensitive matching on Windows filesystems (and to tolerate naming drift).
    ann_map: dict[str, Path] = {}
    for d in sorted([p for p in annotated_root.iterdir() if p.is_dir()]):
        key = _strip_suffixes(d.name).lower()
        ann_map.setdefault(key, d)

    replaced = 0
    missing_ann = 0
    for rdir in rough_dirs:
        indicator = _strip_suffixes(rdir.name).lower()
        ann_dir = ann_map.get(indicator)
        if ann_dir is None:
            missing_ann += 1
            continue

        for ann_fp in sorted([p for p in ann_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]):
            idx = _parse_annot_frame_index(ann_fp.name)
            if idx is None:
                continue
            target = _find_roughmask_target(rdir, idx)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ann_fp, target)
            replaced += 1

        print(f"{rdir.name}: applied annotations from {ann_dir.name}")

    print(f"done: replaced {replaced} roughmask frames")
    if missing_ann:
        print(f"note: {missing_ann} roughmask folders had no matching annotated folder")


def _iter_time_frames(vol, time_axis: int) -> Iterator:
    # NIfTI arrays are often (H, W, T) or (T, H, W). We default to last axis.
    if getattr(vol, "ndim", 0) == 2:
        yield vol
        return
    ax = time_axis if time_axis >= 0 else vol.ndim + time_axis
    if ax < 0 or ax >= vol.ndim:
        raise ValueError(f"time_axis {time_axis} invalid for shape {vol.shape}")
    for i in range(vol.shape[ax]):
        yield i, vol.take(i, axis=ax)


def _label_slice_to_u8(x):
    import numpy as np

    arr = np.asarray(x)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def extract_two_annotations_per_label(
    label_root: Path,
    out_root: Path,
    time_axis: int = -1,
) -> None:
    """
    For each ``*_label.nii.gz`` in ``label_root`` export the first two non-empty
    frames as PNGs into ``out_root/<stem>/frame_XXX.png``.
    """
    try:
        import nibabel as nib
        import numpy as np
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependencies for --annot. Install: pip install nibabel numpy pillow"
        ) from e

    label_root = label_root.resolve()
    out_root = out_root.resolve()

    if not label_root.is_dir():
        raise FileNotFoundError(f"wallMotion label root not found: {label_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(label_root.glob("*_label.nii.gz"), key=lambda p: p.name.lower())
    if not label_paths:
        print(f"no *_label.nii.gz files found in {label_root}")
        return

    for path in label_paths:
        stem = path.name
        stem = os.path.splitext(os.path.splitext(stem)[0])[0]
        case_out = out_root / stem
        case_out.mkdir(parents=True, exist_ok=True)

        try:
            vol = nib.load(str(path)).get_fdata(dtype=np.float32)
        except Exception as e:
            print(f"skip unreadable label volume: {path.name} ({e})")
            continue

        chosen: list[int] = []
        for idx, sl in _iter_time_frames(vol, time_axis=time_axis):
            if len(chosen) >= 2:
                break
            if not np.any(np.isfinite(sl) & (sl != 0)):
                continue
            out_path = case_out / f"frame_{idx:03d}.png"
            Image.fromarray(_label_slice_to_u8(sl), mode="L").save(str(out_path))
            chosen.append(idx)

        if len(chosen) < 2:
            print(f"{path.name} -> {case_out} (only {len(chosen)} annotated frames found: {chosen})")
        else:
            print(f"{path.name} -> {case_out} (2 annotated frames: {chosen})")


def _resolve_sam2_video_root() -> Path:
    """
    Prefer the user's requested convention Preproccessed_data/sam_2_videos, but
    remain compatible with existing Preproccessed_data/sam2_videos.
    """
    env = os.environ.get("SAM2_VIDEO_ROOT")
    if env:
        return Path(env).resolve()

    candidates = [
        _PROJECT / "Preproccessed_data" / "sam_2_videos",
        _PROJECT / "Preproccessed_data" / "sam2_videos",
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return candidates[0].resolve()


def main() -> None:
    p = argparse.ArgumentParser(description="Dataset helpers for LVdisplacement.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--inventory",
        action="store_true",
        help="show inventory for Preproccessed_data\\sam_2_videos (or sam2_videos)",
    )
    mode.add_argument(
        "--annot",
        action="store_true",
        help="extract two annotated frames per *_label.nii.gz into Preproccessed_Data\\Annotated-frames",
    )
    mode.add_argument(
        "--apply-roughmasks",
        action="store_true",
        help="(legacy) apply annotated PNGs onto roughmask folders",
    )
    p.add_argument(
        "--rough-root",
        default=str(DEFAULT_PREPROCCESSED_DATA / "sam2roughmasks"),
        help="root folder containing <indicator>_roughmask/ subfolders",
    )
    p.add_argument(
        "--annotated-root",
        default=str(DEFAULT_PREPROCCESSED_DATA / "Annotated_frames"),
        help="root folder containing annotated frame folders (frame_XXX.png)",
    )
    p.add_argument(
        "--wallmotion-root",
        default=str(DEFAULT_WALLMOTION_ROOT),
        help="root folder containing wallMotion *_label.nii.gz files",
    )
    p.add_argument(
        "--annot-out",
        default=str(DEFAULT_PREPROCCESSED_DATA / "Annotated-frames"),
        help="output root for --annot",
    )
    p.add_argument(
        "--time-axis",
        type=int,
        default=-1,
        help="axis to treat as time for *_label.nii.gz (default -1 = last axis)",
    )
    args = p.parse_args()

    if args.inventory:
        root = _resolve_sam2_video_root()
        if not root.is_dir():
            print(f"Video root does not exist yet: {root}")
            print(
                "Create it and add '{Indicator}_image' folders with JPEG frames, "
                "or set SAM2_VIDEO_ROOT."
            )
            entries: list[VideoEntry] = []
        else:
            entries = discover_sam2_videos(root)
        print_video_inventory(entries, root)
        return

    if args.annot:
        extract_two_annotations_per_label(
            label_root=Path(args.wallmotion_root),
            out_root=Path(args.annot_out),
            time_axis=int(args.time_axis),
        )
        return

    apply_annotations_to_roughmasks(
        rough_root=Path(args.rough_root),
        annotated_root=Path(args.annotated_root),
    )


if __name__ == "__main__":
    main()

