"""
MRI Report Evaluation — IMPROVED
==================================
Changes addressing professor feedback:

  [PROF #1] Reports actual evaluation results with a printed summary table
            and saves evaluation_summary.json with all metrics.

  [PROF #2] Radiologist scoring scaffold:
            - Loads expert_scores.json if it exists (you fill this in with
              radiologist ratings) and incorporates them into the summary.
            - Prints which cases most need expert review.

  [PROF #3] Hallucination risk is included in per-case metrics and summary.

  [PROF #4] Baseline comparison:
            - Computes the same metrics for the rule-based baseline report
            - Prints AI vs Baseline comparison table side by side.

  [PROF #5] contrast_enhancement field from findings_json is now tracked.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(r"D:\Downloads\Agent\data\BraTS2020_training_data\content\data")
OUTPUT_DIR  = Path(r"D:\Downloads\Agent\output\slices")
REPORTS_DIR = Path(r"D:\Downloads\Agent\output\reports")
MANIFEST_PATH       = OUTPUT_DIR / "manifest.json"
CASE_TABLE_PATH     = REPORTS_DIR / "evaluation_case_table.json"
SUMMARY_PATH        = REPORTS_DIR / "evaluation_summary.json"
EXPERT_SCORES_PATH  = REPORTS_DIR / "expert_scores.json"   # [PROF #2] fill this in


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_manifest(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_report_results(reports_dir: Path) -> List[dict]:
    results = []
    for p in sorted(reports_dir.glob("*_result.json")):
        with open(p, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results

def parse_slice_index(name: str) -> int:
    m = re.search(r"_slice_(\d+)", name)
    return int(m.group(1)) if m else 0

def read_h5_mask(h5_path: Path) -> np.ndarray:
    with h5py.File(str(h5_path), "r") as f:
        mask = f["mask"][:]
    combined = np.zeros(mask.shape[:2], dtype=np.uint8)
    for i in range(mask.shape[2]):
        combined[mask[:, :, i] == 1] = i + 1
    return combined

def binary_mask(m: np.ndarray) -> np.ndarray:
    return (m > 0).astype(np.uint8)

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def has_any(text: str, kws: List[str]) -> bool:
    return any(k in text for k in kws)


# ── Ground truth extraction  ──────────────────────────────────────────────────

def classify_burden(pixels: int) -> str:
    # Thresholds scaled for up to 10 slices
    if pixels == 0:        return "none"
    if pixels < 1500:      return "small"
    if pixels < 15000:     return "medium"
    return "large"

def compute_slice_gt(mask_bin: np.ndarray) -> Dict:
    tumor_px = int(mask_bin.sum())
    h, w     = mask_bin.shape
    left_px  = int(mask_bin[:, :w//2].sum())
    right_px = int(mask_bin[:, w//2:].sum())

    if tumor_px == 0:
        lat = "none"
    else:
        ratio = abs(left_px - right_px) / max(tumor_px, 1)
        lat = "bilateral" if ratio < 0.35 else ("left" if left_px > right_px else "right")

    return {
        "tumor_present": tumor_px > 0,
        "tumor_pixels":  tumor_px,
        "laterality":    lat,
        "left_pixels":   left_px,
        "right_pixels":  right_px,
    }

def aggregate_case_gt(volume_id: str, slice_indices: List[int], data_dir: Path) -> Dict:
    per_slice = []
    for idx in slice_indices:
        h5_path = data_dir / f"{volume_id}_slice_{idx}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing: {h5_path}")
        combined = read_h5_mask(h5_path)
        per_slice.append(compute_slice_gt(binary_mask(combined)))

    tumor_any  = any(s["tumor_present"] for s in per_slice)
    total_px   = sum(s["tumor_pixels"] for s in per_slice)
    votes      = {"left": 0, "right": 0, "bilateral": 0}
    for s in per_slice:
        if s["laterality"] in votes:
            votes[s["laterality"]] += 1
    laterality = "none" if not tumor_any else max(votes, key=votes.get)

    return {
        "tumor_present":       tumor_any,
        "tumor_pixels_total":  total_px,
        "tumor_burden":        classify_burden(total_px),
        "laterality":          laterality,
        "positive_slice_count": sum(1 for s in per_slice if s["tumor_present"]),
        "num_slices":          len(per_slice),
    }


# ── Prediction extraction  ────────────────────────────────────────────────────

def extract_laterality(text: str) -> str:
    t = text.lower()
    if has_any(t, ["bilateral", "both hemispheres", "both sides"]): return "bilateral"
    if has_any(t, ["left hemisphere", "left-sided", "left side",
                   "left frontal", "left parietal", "left temporal",
                   "left occipital", "left posterior", "left anterior"]): return "left"
    if has_any(t, ["right hemisphere", "right-sided", "right side",
                   "right frontal", "right parietal", "right temporal",
                   "right occipital", "right posterior", "right anterior"]): return "right"
    return "unclear"

def extract_burden(text: str) -> str:
    t = text.lower()
    if has_any(t, ["large", "extensive", "marked", "substantial", "massive"]): return "large"
    if has_any(t, ["small", "tiny", "focal", "limited", "subtle"]):           return "small"
    if has_any(t, ["moderate", "well-circumscribed", "surrounding edema"]):   return "medium"
    return "unclear"

def extract_prediction(result_json: dict) -> Dict:
    text = "\n".join([
        result_json.get("observations", ""),
        result_json.get("analysis", ""),
        result_json.get("report", ""),
    ]).lower()

    fj = result_json.get("findings_json", {})
    lat = fj.get("laterality", "unclear")
    refined = extract_laterality(text)
    if refined != "unclear":
        lat = refined

    return {
        "tumor_present":           bool(fj.get("tumor_present", False)),
        "mass_effect":             bool(fj.get("mass_effect", False)),
        "midline_shift":           bool(fj.get("midline_shift", False)),
        "ventricular_abnormality": bool(fj.get("ventricular_abnormality", False)),
        "hyperintense_region":     bool(fj.get("hyperintense_region", False)),
        "contrast_enhancement":    bool(fj.get("contrast_enhancement", False)),  # [PROF #5]
        "laterality":              lat,
        "confidence":              fj.get("confidence", "unclear"),
        "predicted_burden":        extract_burden(text),
        "modalities_reviewed":     fj.get("modalities_reviewed", ["flair"]),     # [PROF #5]
    }


# ── Report quality scoring  ───────────────────────────────────────────────────

def score_quality(report_text: str) -> Dict:
    """12-point rubric (was 8). Higher = better report."""
    t = report_text.lower()
    checks = {
        "has_findings":         "findings:"        in t,
        "has_impression":       "impression:"      in t,
        "has_recommendations":  "recommendations:" in t,
        "has_confidence":       "confidence:"      in t,
        "has_sequences":        "sequences reviewed:" in t,           # [PROF #5]
        "mentions_signal":      has_any(t, ["lesion", "hyperintense", "hypointense",
                                            "mass", "neoplasm", "edema"]),
        "mentions_location":    has_any(t, ["left", "right", "bilateral",
                                            "frontal", "parietal", "temporal"]),
        "mentions_structure":   has_any(t, ["ventricle", "white matter",
                                            "brain parenchyma", "midline"]),
        "mentions_enhancement": has_any(t, ["enhancement", "t1ce", "contrast"]),  # [PROF #5]
        "mentions_differential":has_any(t, ["differential", "consider", "consistent with",
                                            "may represent", "cannot exclude"]),
        "mentions_followup":    has_any(t, ["follow-up", "recommend", "correlation"]),
        "reasonable_length":    150 <= len(report_text) <= 3500,
    }
    score = sum(int(v) for v in checks.values()) / len(checks)
    return {"quality_score": round(score, 4), "quality_checks": checks}


# ── Metrics  ──────────────────────────────────────────────────────────────────

def classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    return {
        "n_cases":   len(y_true),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":  round(safe_div(tp + tn, len(y_true)), 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(safe_div(2 * precision * recall, precision + recall), 4),
    }

def laterality_soft(gt: str, pred: str) -> float:
    if gt == pred: return 1.0
    if gt in {"left", "right"} and pred == "bilateral": return 0.5
    if gt == "bilateral" and pred in {"left", "right"}: return 0.5
    return 0.0


# ── [PROF #2] Radiologist scoring  ───────────────────────────────────────────

def load_expert_scores(path: Path) -> Dict:
    """
    Load radiologist expert scores if file exists.
    Format expected in expert_scores.json:
    [
      {
        "volume_id": "volume_1",
        "accuracy":   4,        (1-5 Likert: 1=completely wrong, 5=excellent)
        "completeness": 3,
        "clinical_usefulness": 4,
        "would_use_as_draft": true,
        "notes": "Correctly identified right frontal lesion"
      }, ...
    ]
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {r["volume_id"]: r for r in rows}

def summarize_expert_scores(expert_lookup: Dict) -> Dict:
    if not expert_lookup:
        return {}
    scores = list(expert_lookup.values())
    return {
        "n_expert_reviewed":         len(scores),
        "mean_accuracy":             round(np.mean([s.get("accuracy",0) for s in scores]), 2),
        "mean_completeness":         round(np.mean([s.get("completeness",0) for s in scores]), 2),
        "mean_clinical_usefulness":  round(np.mean([s.get("clinical_usefulness",0) for s in scores]), 2),
        "pct_would_use_as_draft":    round(
            sum(1 for s in scores if s.get("would_use_as_draft")) / len(scores) * 100, 1
        ),
    }


# ── [PROF #1] Print results table  ───────────────────────────────────────────

def print_results_table(case_rows: List[dict]):
    """[PROF #1] Print an actual readable results table."""
    header = (f"{'Volume':<14} {'GT tumor':>8} {'AI tumor':>8} {'Match':>6} "
              f"{'GT lat':>8} {'AI lat':>8} {'Lat✓':>5} "
              f"{'Risk':>7} {'Quality':>8}")
    print("\n" + "=" * len(header))
    print("PER-CASE EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for row in case_rows:
        match_str = "✅" if row["tumor_presence_match"] else "❌"
        lat_str   = "✅" if row["laterality_match"] else "❌"
        risk      = row.get("hallucination_risk", "?")
        risk_str  = {"low": "LOW ", "medium": "MED ", "high": "HIGH", "unknown": "? "}.get(risk, risk)
        flag      = " ⚠" if row.get("flagged_for_review") else ""
        print(
            f"{row['volume_id']:<14} "
            f"{str(row['gt_tumor_present']):>8} "
            f"{str(row['pred_tumor_present']):>8} "
            f"{match_str:>6} "
            f"{row['gt_laterality']:>8} "
            f"{row['pred_laterality']:>8} "
            f"{lat_str:>5} "
            f"{risk_str:>7}{flag} "
            f"{row['quality_score']:>8.2f}"
        )
    print("=" * len(header))


def print_comparison_table(ai_metrics: Dict, baseline_metrics: Dict):
    """[PROF #4] Print AI vs Baseline side-by-side comparison."""
    print("\n" + "=" * 55)
    print("AI vs BASELINE COMPARISON  [PROF #4]")
    print("=" * 55)
    print(f"{'Metric':<25} {'AI Model':>12} {'Baseline':>12}")
    print("-" * 55)
    for key in ["accuracy", "precision", "recall", "f1", "avg_quality_score"]:
        ai_val  = ai_metrics.get(key, "N/A")
        bas_val = baseline_metrics.get(key, "N/A")
        better  = ""
        if isinstance(ai_val, float) and isinstance(bas_val, float):
            better = " ◀" if ai_val > bas_val else ("  " if ai_val == bas_val else "  ")
        print(f"{key:<25} {str(ai_val):>12} {str(bas_val):>12}{better}")
    print("=" * 55)
    print("◀ = AI outperforms baseline")


# ── Main evaluation  ──────────────────────────────────────────────────────────

def evaluate_all():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    results      = load_report_results(REPORTS_DIR)
    expert_lookup = load_expert_scores(EXPERT_SCORES_PATH)   # [PROF #2]

    if not results:
        print("No *_result.json files found.")
        return

    print(f"\nEvaluating {len(results)} reports...")

    case_rows = []
    # AI tracking
    ai_y_true, ai_y_pred = [], []
    ai_quality = []
    ai_lat_exact, ai_lat_soft = 0, []
    ai_halluc_counts = {"low": 0, "medium": 0, "high": 0, "unknown": 0}

    # Baseline tracking  [PROF #4]
    bl_y_pred  = []
    bl_quality = []

    for result_json in results:
        volume_id = result_json["volume_id"]
        slice_indices = [parse_slice_index(x) for x in result_json["slices_used"]]

        try:
            gt = aggregate_case_gt(volume_id, slice_indices, DATA_DIR)
        except FileNotFoundError as e:
            print(f"  SKIP {volume_id}: {e}")
            continue

        pred    = extract_prediction(result_json)
        quality = score_quality(result_json.get("report", ""))

        # Baseline prediction is always "no tumor"  [PROF #4]
        baseline_quality = score_quality(result_json.get("baseline_report", ""))
        bl_y_pred.append(0)   # baseline always predicts no tumor
        bl_quality.append(baseline_quality["quality_score"])

        # AI metrics
        ai_y_true.append(int(gt["tumor_present"]))
        ai_y_pred.append(int(pred["tumor_present"]))
        ai_quality.append(quality["quality_score"])

        lat_match = gt["laterality"] == pred["laterality"]
        lat_soft  = laterality_soft(gt["laterality"], pred["laterality"])
        if lat_match:
            ai_lat_exact += 1
        ai_lat_soft.append(lat_soft)

        risk = result_json.get("hallucination_risk", "unknown")
        ai_halluc_counts[risk] = ai_halluc_counts.get(risk, 0) + 1

        expert_row = expert_lookup.get(volume_id, {})  # [PROF #2]

        row = {
            "volume_id":              volume_id,
            "selected_slices":        slice_indices,
            "modality_used":          result_json.get("modality_used", "flair_only"),

            # Ground truth
            "gt_tumor_present":       gt["tumor_present"],
            "gt_positive_slices":     gt["positive_slice_count"],
            "gt_tumor_pixels":        gt["tumor_pixels_total"],
            "gt_tumor_burden":        gt["tumor_burden"],
            "gt_laterality":          gt["laterality"],

            # AI prediction
            "pred_tumor_present":     pred["tumor_present"],
            "pred_mass_effect":       pred["mass_effect"],
            "pred_midline_shift":     pred["midline_shift"],
            "pred_ventricular_abnorm":pred["ventricular_abnormality"],
            "pred_contrast_enhancement": pred["contrast_enhancement"],   # [PROF #5]
            "pred_laterality":        pred["laterality"],
            "pred_confidence":        pred["confidence"],
            "pred_burden":            pred["predicted_burden"],
            "modalities_reviewed":    pred["modalities_reviewed"],       # [PROF #5]

            # Scores
            "quality_score":          quality["quality_score"],
            "quality_checks":         quality["quality_checks"],
            "hallucination_risk":     risk,                              # [PROF #3]
            "flagged_for_review":     result_json.get("flagged_for_review", False),

            # Match flags
            "tumor_presence_match":   gt["tumor_present"] == pred["tumor_present"],
            "laterality_match":       lat_match,
            "laterality_soft_score":  lat_soft,

            # Expert scores  [PROF #2]
            "expert_accuracy":        expert_row.get("accuracy"),
            "expert_completeness":    expert_row.get("completeness"),
            "expert_usefulness":      expert_row.get("clinical_usefulness"),
            "expert_would_use":       expert_row.get("would_use_as_draft"),
            "expert_notes":           expert_row.get("notes", ""),
        }
        case_rows.append(row)

    # ── Compute final metrics ──────────────────────────────────────────────────
    ai_metrics = classification_metrics(ai_y_true, ai_y_pred)
    ai_metrics["avg_quality_score"]       = round(float(np.mean(ai_quality)), 4)
    ai_metrics["laterality_accuracy"]     = round(safe_div(ai_lat_exact, len(case_rows)), 4)
    ai_metrics["avg_laterality_soft"]     = round(float(np.mean(ai_lat_soft)), 4)
    ai_metrics["hallucination_breakdown"] = ai_halluc_counts                  # [PROF #3]
    ai_metrics["pct_flagged"]             = round(
        sum(1 for r in case_rows if r["flagged_for_review"]) / max(len(case_rows), 1) * 100, 1
    )
    ai_metrics["expert_summary"]          = summarize_expert_scores(expert_lookup)  # [PROF #2]
    ai_metrics["multimodal_cases"]        = sum(
        1 for r in case_rows if r["modality_used"] == "multimodal_4_sequences"
    )                                                                          # [PROF #5]

    # Baseline metrics  [PROF #4]
    bl_metrics = classification_metrics(ai_y_true, bl_y_pred)
    bl_metrics["avg_quality_score"] = round(float(np.mean(bl_quality)), 4)

    # ── [PROF #1] Print actual results ──────────────────────────────────────
    print_results_table(case_rows)

    print("\n" + "=" * 55)
    print("AGGREGATE METRICS — AI MODEL")
    print("=" * 55)
    print(json.dumps({k: v for k, v in ai_metrics.items()
                      if k != "expert_summary"}, indent=2))

    print_comparison_table(ai_metrics, bl_metrics)  # [PROF #4]

    if expert_lookup:                               # [PROF #2]
        print("\n" + "=" * 55)
        print("EXPERT RADIOLOGIST EVALUATION  [PROF #2]")
        print("=" * 55)
        print(json.dumps(ai_metrics["expert_summary"], indent=2))
    else:
        print("\n[PROF #2] No expert_scores.json found.")
        print("  → Fill in output/reports/expert_scores.json with radiologist ratings.")
        print("  → Template: [{volume_id, accuracy(1-5), completeness(1-5),")
        print("               clinical_usefulness(1-5), would_use_as_draft(bool), notes}]")

    # ── Save outputs ──────────────────────────────────────────────────────────
    summary = {
        "ai_metrics":       ai_metrics,
        "baseline_metrics": bl_metrics,   # [PROF #4]
        "n_cases":          len(case_rows),
        "generated_at":     str(Path(__file__).stat().st_mtime),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(CASE_TABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(case_rows, f, indent=2)

    print(f"\n  Saved → {SUMMARY_PATH.name}")
    print(f"  Saved → {CASE_TABLE_PATH.name}")

    # ── [PROF #2] Print top candidates for expert review ──────────────────
    flagged = [r for r in case_rows if r["flagged_for_review"]]
    if flagged:
        print(f"\n  ⚠  {len(flagged)} cases flagged for radiologist review:")
        for r in flagged[:10]:
            print(f"     {r['volume_id']}  risk={r['hallucination_risk']}")


if __name__ == "__main__":
    evaluate_all()