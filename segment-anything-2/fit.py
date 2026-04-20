"""
Batch-run SAM2 on every frame folder under Preproccessed_data/sam2_videos.

Input folder convention: {indicator}_image
Output folder convention: {indicator}_prediction

Pipeline (default):
  1) Use pre-generated rough masks from sam2roughmasks/ (no generation here)
  2) Merge annotated frame PNGs into rough masks (replacement) as GT anchors
  3) Run SAM2 video predictor using the merged rough masks as per-frame mask prompts
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Script location: .../LVdisplacement/segment-anything-2/NIGGA.py
_REPO_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _REPO_ROOT.parent

# Ensure imports from sibling project files and from sam2 package work.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extractname import apply_annotations_to_roughmasks, discover_sam2_videos

def _resolve_checkpoint() -> str:
    candidates = [
        _PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_large.pt",
        _REPO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        "Could not find sam2.1_hiera_large.pt. Place it in "
        "LVdisplacement/checkpoints/ or segment-anything-2/checkpoints/, "
        "or set SAM2_CHECKPOINT."
    )


def _resolve_data_roots() -> tuple[Path, Path]:
    video_candidates = [
        _PROJECT_ROOT / "Preprocecessed_Data" / "sam2_videos",
        _PROJECT_ROOT / "Preprocessed_Data" / "sam2_videos",
        _PROJECT_ROOT / "Preproccessed_data" / "sam2_videos",
    ]
    pred_candidates = [
        _PROJECT_ROOT / "Preprocecessed_Data" / "sam2_predictions",
        _PROJECT_ROOT / "Preprocessed_Data" / "sam2_predictions",
        _PROJECT_ROOT / "Preproccessed_data" / "sam2_predictions",
    ]
    video_root = next((p for p in video_candidates if p.is_dir()), video_candidates[0])
    pred_root = next((p for p in pred_candidates if p.parent.is_dir()), pred_candidates[0])
    return video_root, pred_root


def _resolve_rough_and_annot_roots() -> tuple[Path, Path]:
    rough_candidates = [
        _PROJECT_ROOT / "Preproccessed_data" / "sam2roughmasks",
        _PROJECT_ROOT / "Preproccessed_Data" / "sam2roughmasks",
        _PROJECT_ROOT / "Preprocessed_Data" / "sam2roughmasks",
    ]
    annot_candidates = [
        # new (extractname.py --annot) output
        _PROJECT_ROOT / "Preprocessed_Data" / "Annotated-frames",
        # legacy output
        _PROJECT_ROOT / "Preproccessed_Data" / "Annotated_frames",
        _PROJECT_ROOT / "Preproccessed_data" / "Annotated_frames",
        _PROJECT_ROOT / "Preproccessed_data" / "Annotated-frames",
    ]
    rough_root = next((p for p in rough_candidates if p.parent.is_dir()), rough_candidates[0])
    annot_root = next((p for p in annot_candidates if p.parent.is_dir()), annot_candidates[0])
    return rough_root, annot_root


def _prediction_folder_name(folder_name: str) -> str:
    if folder_name.endswith("_image"):
        return f"{folder_name[:-len('_image')]}_prediction"
    return f"{folder_name}_prediction"


def _frame_paths(video_dir: Path) -> list[Path]:
    frames = [
        p
        for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
    ]
    frames.sort(key=lambda p: int(p.stem))
    return frames


def _indicator_from_video_folder(folder_name: str) -> str:
    return folder_name[:-len("_image")] if folder_name.endswith("_image") else folder_name


def _roughmask_dir_name(indicator: str) -> str:
    # Support both conventions.
    return f"{indicator}_roughmask"


def _resolve_rough_dir(rough_root: Path, indicator: str) -> Path:
    """
    Roughmask folders might be named either:
      - <indicator>_roughmask
      - <indicator>_roughmasks
    Prefer whichever exists.
    """
    a = rough_root / f"{indicator}_roughmask"
    b = rough_root / f"{indicator}_roughmasks"
    if a.is_dir():
        return a
    if b.is_dir():
        return b
    # default to singular (for error messages / creation elsewhere)
    return a


def _read_mask_png_bool(path: Path, size_hw: tuple[int, int]) -> np.ndarray:
    """
    Read a roughmask/annotation PNG as a boolean HxW mask.
    If the size doesn't match, resize with nearest-neighbor.
    """
    m = Image.open(path).convert("L")
    if m.size != (size_hw[1], size_hw[0]):
        m = m.resize((size_hw[1], size_hw[0]), resample=Image.Resampling.NEAREST)
    arr = np.array(m)
    return arr > 0


def _best_annotation(anns: list[dict], h: int, w: int) -> np.ndarray:
    # Keep only one mask: the annotation with highest predicted_iou.
    if not anns:
        return np.zeros((h, w), dtype=np.uint8)

    best_seg: np.ndarray | None = None
    best_score = float("-inf")
    for ann in anns:
        seg = ann.get("segmentation")
        if not (isinstance(seg, np.ndarray) and seg.ndim == 2):
            continue
        score = float(ann.get("predicted_iou", float("-inf")))
        if score > best_score:
            best_score = score
            best_seg = seg

    if best_seg is None:
        return np.zeros((h, w), dtype=np.uint8)
    return (best_seg.astype(bool).astype(np.uint8) * 255)


def _build_generator(device: torch.device) -> SAM2AutomaticMaskGenerator:
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "SAM2 dependencies not installed in this Python env. "
            "From segment-anything-2/, install requirements (hydra-core, etc.) "
            "and ensure you're using that environment."
        ) from e

    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    ckpt = os.environ.get("SAM2_CHECKPOINT", _resolve_checkpoint())
    sam2_model = build_sam2(model_cfg, ckpt, device=device)
    return SAM2AutomaticMaskGenerator(
        model=sam2_model,
        output_mode="binary_mask",
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.95,
    )


def _build_video_predictor(device: torch.device):
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "SAM2 video predictor dependencies not installed in this Python env. "
            "From segment-anything-2/, install requirements (hydra-core, etc.) "
            "and ensure you're using that environment."
        ) from e

    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    ckpt = os.environ.get("SAM2_CHECKPOINT", _resolve_checkpoint())
    return build_sam2_video_predictor(model_cfg, ckpt, device=device)


@torch.inference_mode()
def run_sam2_on_all_videos(video_root: Path | None = None, pred_root: Path | None = None) -> None:
    default_video_root, default_pred_root = _resolve_data_roots()
    video_root = (video_root or default_video_root).resolve()
    pred_root = (pred_root or default_pred_root).resolve()
    if not video_root.is_dir():
        raise FileNotFoundError(f"missing video folder: {video_root}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nMPS support is preliminary. SAM2 was trained with CUDA and can differ on MPS."
        )

    generator = _build_generator(device)
    entries = discover_sam2_videos(video_root)
    if not entries:
        print(f"no frame folders found in {video_root}")
        return

    pred_root.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        frames = _frame_paths(entry.root)
        if not frames:
            continue
        out_dir = pred_root / _prediction_folder_name(entry.folder_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        for fp in frames:
            img = np.array(Image.open(fp).convert("RGB"))
            h, w = img.shape[:2]
            anns = generator.generate(img)
            mask = _best_annotation(anns, h, w)
            Image.fromarray(mask, mode="L").save(out_dir / f"{fp.stem}.png")

        print(f"{entry.folder_name} -> {out_dir} ({len(frames)} frames)")


@torch.inference_mode()
def run_sam2_video_with_mask_prompts(
    video_root: Path,
    rough_root: Path,
    pred_root: Path,
    prompt_every_frame: bool = True,
) -> None:
    """
    For each video folder, initialize SAM2 video predictor and add mask prompts from
    `<rough_root>/<indicator>_roughmask/`.

    If `prompt_every_frame` is True, it adds a mask prompt on every frame (slow but
    matches the user's request). Otherwise, it only seeds on the first frame.
    """
    if not video_root.is_dir():
        raise FileNotFoundError(f"missing video folder: {video_root}")
    if not rough_root.is_dir():
        raise FileNotFoundError(f"missing roughmask folder: {rough_root}")

    pred_root.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    predictor = _build_video_predictor(device)
    entries = discover_sam2_videos(video_root)
    if not entries:
        print(f"no frame folders found in {video_root}")
        return

    for entry in entries:
        indicator = _indicator_from_video_folder(entry.folder_name)
        rough_dir = _resolve_rough_dir(rough_root, indicator)
        if not rough_dir.is_dir():
            print(f"skip {entry.folder_name}: missing roughmask dir {rough_dir}")
            continue

        frames = _frame_paths(entry.root)
        if not frames:
            continue

        out_dir = pred_root / _prediction_folder_name(entry.folder_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        state = predictor.init_state(video_path=str(entry.root))
        obj_id = 1

        # Add mask prompts.
        seeded = False
        for i, fp in enumerate(frames):
            mpath = rough_dir / f"{fp.stem}.png"
            if not mpath.is_file():
                continue
            img0 = Image.open(fp).convert("RGB")
            h, w = img0.size[1], img0.size[0]
            mask_bool = _read_mask_png_bool(mpath, size_hw=(h, w))
            if not np.any(mask_bool):
                continue

            predictor.add_new_mask(state, frame_idx=i, obj_id=obj_id, mask=mask_bool)
            seeded = True
            if not prompt_every_frame:
                break

        if not seeded:
            print(f"skip {entry.folder_name}: no non-empty roughmask prompts found")
            continue

        # Propagate and write predictions.
        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state):
            # Take obj_id=1 (index 0).
            m = video_res_masks[0, 0]
            m_np = (m > 0).to(torch.uint8).detach().cpu().numpy() * 255
            Image.fromarray(m_np, mode="L").save(out_dir / f"{frames[frame_idx].stem}.png")

        print(f"{entry.folder_name} -> {out_dir} (mask-prompted SAM2 video)")


if __name__ == "__main__":
    # Default behavior: prompt-based pipeline using *pre-generated* roughmasks.
    # You can override with env vars:
    #   SAM2_STAGE=merge|prompted|auto
    stage = os.environ.get("SAM2_STAGE", "prompted").strip().lower()

    video_root, pred_root = _resolve_data_roots()
    rough_root, annot_root = _resolve_rough_and_annot_roots()

    if stage == "auto":
        run_sam2_on_all_videos(video_root=video_root, pred_root=pred_root)
        raise SystemExit(0)

    if stage in ("merge", "prompted"):
        # Replace roughmask frames with any annotated frame PNGs when available.
        apply_annotations_to_roughmasks(rough_root=rough_root, annotated_root=annot_root)

    if stage == "merge":
        raise SystemExit(0)

    # stage == prompted (default)
    prompt_every = os.environ.get("PROMPT_EVERY_FRAME", "1").strip() == "1"
    run_sam2_video_with_mask_prompts(
        video_root=video_root,
        rough_root=rough_root,
        pred_root=pred_root,
        prompt_every_frame=prompt_every,
    )