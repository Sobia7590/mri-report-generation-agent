"""
MRI Data Loading & Slice Extraction Pipeline  — IMPROVED
=========================================================
Changes from original:
  [PROF #5] Saves ALL 4 MRI channels (FLAIR, T1, T1ce, T2) as individual PNGs
            AND a combined 4-panel image per slice for multi-modal agent input.
  [PROF #6] Saves ground-truth tumor mask overlay PNG per slice for explainability.

Compatible with: Kaggle BraTS 2020 — HDF5 format
"""

import h5py
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import re

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"D:\Downloads\Agent\data\BraTS2020_training_data\content\data")
OUTPUT_DIR = Path(r"D:\Downloads\Agent\output\slices")
CHANNEL_MAP = {
    "flair": 0,
    "t1":    1,
    "t1ce":  2,
    "t2":    3,
}

# Tumor region colors for overlay (region 1=necrotic, 2=edema, 3=enhancing)
TUMOR_COLORS = {
    1: (255, 50,  50,  140),   # red   — necrotic core
    2: (255, 200, 50,  140),   # yellow — edema
    3: (50,  255, 50,  140),   # green  — enhancing tumor
}

# ──────────────────────────────────────────────────────────────────────────────

def normalize_channel(arr: np.ndarray) -> np.ndarray:
    """Normalize a 2D float array to uint8 [0-255]."""
    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    p1, p99 = np.percentile(nonzero, [1, 99])
    arr = np.clip(arr, p1, p99)
    arr = (arr - p1) / (p99 - p1 + 1e-8)
    return (arr * 255).astype(np.uint8)


def read_h5_slice(h5_path: Path) -> dict:
    """Read one .h5 slice — returns all 4 channels + raw mask."""
    with h5py.File(str(h5_path), "r") as f:
        image = f["image"][:]   # (240, 240, 4)
        mask  = f["mask"][:]    # (240, 240, 3)

    channels = {}
    for name, idx in CHANNEL_MAP.items():
        channels[name] = normalize_channel(image[:, :, idx])

    # Combined label map: 0=bg, 1=necrotic, 2=edema, 3=enhancing
    combined_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for i in range(mask.shape[2]):
        combined_mask[mask[:, :, i] == 1] = i + 1
    channels["mask"] = combined_mask
    return channels


def get_volume_id(h5_path: Path) -> tuple:
    match = re.match(r"(volume_\d+)_slice_(\d+)", h5_path.stem)
    if match:
        return match.group(1), int(match.group(2))
    return h5_path.stem, 0


def save_png(arr: np.ndarray, path: Path):
    Image.fromarray(arr).save(str(path))


def save_mask_overlay(base_arr: np.ndarray, mask_arr: np.ndarray, path: Path):
    """
    [PROF #6] Save a FLAIR image with colored tumor region overlay.
    Regions: red=necrotic, yellow=edema, green=enhancing tumor.
    """
    base_rgb = Image.fromarray(base_arr).convert("RGBA")
    overlay  = Image.new("RGBA", base_rgb.size, (0, 0, 0, 0))
    draw     = ImageDraw.Draw(overlay)

    for region_id, color in TUMOR_COLORS.items():
        ys, xs = np.where(mask_arr == region_id)
        for x, y in zip(xs, ys):
            draw.point((x, y), fill=color)

    composite = Image.alpha_composite(base_rgb, overlay).convert("RGB")
    composite.save(str(path))


def save_multimodal_panel(channels: dict, slice_idx: int, path: Path):
    """
    [PROF #5] Save a single 4-panel image with all sequences side by side.
    This is what gets sent to Claude instead of FLAIR-only.
    """
    labels = ["FLAIR", "T1", "T1ce", "T2"]
    ch_keys = ["flair", "t1", "t1ce", "t2"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), facecolor="#0f0f0f")
    for ax, key, label in zip(axes, ch_keys, labels):
        ax.imshow(channels[key], cmap="gray")
        ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=4)
        ax.axis("off")

    fig.suptitle(f"Slice {slice_idx:03d} — All Sequences",
                 color="#cccccc", fontsize=10, y=1.02)
    plt.tight_layout(pad=0.3)
    fig.savefig(str(path), dpi=120, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close(fig)


def process_volume(volume_id: str, h5_files: list, output_root: Path) -> dict:
    """
    Process all slices for one volume.
    Saves per slice:
      - flair_slice_XXX.png          (original, for compatibility)
      - t1_slice_XXX.png             [NEW — PROF #5]
      - t1ce_slice_XXX.png           [NEW — PROF #5]
      - t2_slice_XXX.png             [NEW — PROF #5]
      - multimodal_slice_XXX.png     [NEW — PROF #5] 4-panel combined image
      - overlay_slice_XXX.png        [NEW — PROF #6] GT mask overlay
    """
    vol_out    = output_root / volume_id
    slices_dir = vol_out / "slices"
    overlay_dir = vol_out / "overlays"     # [PROF #6]
    multi_dir   = vol_out / "multimodal"   # [PROF #5]

    for d in [vol_out, slices_dir, overlay_dir, multi_dir]:
        d.mkdir(parents=True, exist_ok=True)

    h5_files_sorted = sorted(h5_files, key=lambda p: get_volume_id(p)[1])
    total = len(h5_files_sorted)

    metadata = {
        "volume_id":    volume_id,
        "total_slices": total,
        "slices":       []
    }

    for h5_path in h5_files_sorted:
        _, slice_idx = get_volume_id(h5_path)
        channels     = read_h5_slice(h5_path)

        # ── Original FLAIR PNG (keeps existing agent compatible) ──
        flair_path = slices_dir / f"flair_slice_{slice_idx:03d}.png"
        save_png(channels["flair"], flair_path)

        # ── All 4 individual channel PNGs  [PROF #5] ──
        for ch in ["t1", "t1ce", "t2"]:
            save_png(channels[ch], slices_dir / f"{ch}_slice_{slice_idx:03d}.png")

        # ── 4-panel multimodal image  [PROF #5] ──
        multi_path = multi_dir / f"multimodal_slice_{slice_idx:03d}.png"
        save_multimodal_panel(channels, slice_idx, multi_path)

        # ── GT mask overlay on FLAIR  [PROF #6] ──
        overlay_path = overlay_dir / f"overlay_slice_{slice_idx:03d}.png"
        save_mask_overlay(channels["flair"], channels["mask"], overlay_path)

        metadata["slices"].append({
            "slice_index":      slice_idx,
            "flair_file":       str(flair_path.relative_to(output_root)),
            "multimodal_file":  str(multi_path.relative_to(output_root)),
            "overlay_file":     str(overlay_path.relative_to(output_root)),
            "h5_source":        h5_path.name,
            "has_tumor":        bool(channels["mask"].sum() > 0),
        })

    # ── Comparison grid (unchanged) ──
    compare_dir = vol_out / "comparison"
    compare_dir.mkdir(exist_ok=True)
    n_preview = min(5, total)
    preview_indices = np.linspace(0, total - 1, n_preview, dtype=int)
    preview_files   = [h5_files_sorted[i] for i in preview_indices]
    seq_labels      = ["FLAIR", "T1", "T1ce", "T2"]

    fig, axes = plt.subplots(n_preview, 4,
                             figsize=(16, 3.5 * n_preview),
                             facecolor="#0f0f0f")
    if n_preview == 1:
        axes = axes[np.newaxis, :]

    for row_i, h5_path in enumerate(preview_files):
        _, slice_idx = get_volume_id(h5_path)
        channels     = read_h5_slice(h5_path)
        for col_i, (ch_name, label) in enumerate(zip(CHANNEL_MAP.keys(), seq_labels)):
            ax = axes[row_i][col_i]
            ax.imshow(channels[ch_name], cmap="gray")
            if row_i == 0:
                ax.set_title(label, color="white", fontsize=12,
                             fontweight="bold", pad=5)
            ax.set_ylabel(f"sl {slice_idx}", color="#aaaaaa",
                          fontsize=8, rotation=0, labelpad=28)
            ax.axis("off")

    fig.suptitle(f"{volume_id}  —  Multi-sequence Preview",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout(pad=0.4)
    comp_path = compare_dir / "comparison_grid.png"
    fig.savefig(str(comp_path), dpi=130, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close(fig)

    metadata["comparison_image"] = str(comp_path.relative_to(output_root))
    return metadata


def run_pipeline(data_dir: Path, output_dir: Path, max_volumes: int = None):
    if not data_dir.exists():
        print(f"\nERROR: Data directory not found: {data_dir}\n")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    all_h5 = sorted(data_dir.glob("volume_*_slice_*.h5"))
    if not all_h5:
        print(f"\nERROR: No .h5 files found in {data_dir}\n")
        return []

    volume_map: dict[str, list] = {}
    for h5_path in all_h5:
        vol_id, _ = get_volume_id(h5_path)
        volume_map.setdefault(vol_id, []).append(h5_path)

    volume_ids = sorted(volume_map.keys())
    if max_volumes:
        volume_ids = volume_ids[:max_volumes]

    print(f"\n{'='*55}")
    print(f"  MRI Slice Extraction Pipeline — IMPROVED")
    print(f"  Saves: FLAIR + T1 + T1ce + T2 + multimodal + overlays")
    print(f"  Volumes : {len(volume_ids)}")
    print(f"{'='*55}\n")

    all_metadata = []
    for vol_id in tqdm(volume_ids, desc="Processing volumes"):
        try:
            meta = process_volume(vol_id, volume_map[vol_id], output_dir)
            all_metadata.append(meta)
            tqdm.write(f"  OK  {vol_id}: {meta['total_slices']} slices")
        except Exception as e:
            tqdm.write(f"  ERR {vol_id}: {e}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n  Done! {len(all_metadata)} volumes → {manifest_path}\n")
    return all_metadata


if __name__ == "__main__":
    run_pipeline(DATA_DIR, OUTPUT_DIR, max_volumes=None)