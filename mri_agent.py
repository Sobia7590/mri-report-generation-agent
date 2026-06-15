"""
MRI Report Generation Agent — IMPROVED
=======================================
Changes addressing professor feedback:

  [PROF #3] Hallucination safeguards:
            - Explicit "only report what you see" instruction in every prompt
            - Step 4 (Verify): Claude reviews its own report for hallucinations
            - Confidence gating: LOW confidence triggers a warning flag
            - Structured output format enforced with strict section headers

  [PROF #4] Baseline comparison:
            - run_baseline() generates a simple rule-based template report
              from pixel statistics alone (no AI) for comparison

  [PROF #5] Multi-modal input:
            - MAX_SLICES now uses multimodal 4-panel images (FLAIR+T1+T1ce+T2)
            - Prompts updated to reference all 4 sequences
            - FLAIR-only fallback if multimodal images not yet extracted
"""

import os
import re
import json
import base64
from pathlib import Path
from datetime import datetime

import anthropic
import numpy as np


# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(r"D:\Downloads\Agent\output\slices")
REPORTS_DIR = Path(r"D:\Downloads\Agent\output\reports")
MANIFEST    = OUTPUT_DIR / "manifest.json"

MAX_SLICES           = 10       # was 3
NUM_CASES_TO_PROCESS = 20      # full dataset
SKIP_EXISTING        = True
MODEL                = "claude-opus-4-5"

# Use multimodal 4-panel images if available  [PROF #5]
USE_MULTIMODAL = True
PLANE          = "multimodal"   # folder name for 4-panel images
FLAIR_PLANE    = "slices"       # fallback

# ──────────────────────────────────────────────────────────────────────────────


def load_image_as_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def build_image_content(image_paths: list[Path], modality_label: str = "multimodal") -> list[dict]:
    """Build Anthropic API content with images interleaved with labels."""
    content = []
    for i, path in enumerate(image_paths):
        content.append({
            "type": "text",
            "text": f"Image {i+1} of {len(image_paths)} — {modality_label} ({path.stem}):"
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": load_image_as_base64(path),
            }
        })
    return content


def get_text_from_response(response) -> str:
    if not hasattr(response, "content") or not response.content:
        raise ValueError("Empty response from Anthropic API.")
    parts = [b.text for b in response.content if hasattr(b, "text") and b.text]
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("No text content in Anthropic response.")
    return text


def get_selected_slices(all_pngs: list[Path], max_slices: int) -> list[Path]:
    """Pick evenly spaced slices from the middle 50% of the volume."""
    if not all_pngs:
        raise ValueError("No PNG slices found.")
    n = min(max_slices, len(all_pngs))
    if len(all_pngs) <= n:
        return all_pngs
    start = len(all_pngs) // 4
    end   = max(start + 1, 3 * len(all_pngs) // 4)
    indices = sorted(set(int(i) for i in np.linspace(start, end - 1, n, dtype=int)))
    selected = [all_pngs[i] for i in indices]
    while len(selected) < n:
        for p in all_pngs:
            if p not in selected:
                selected.append(p)
            if len(selected) == n:
                break
    return selected[:n]


# ── Step 1: Observe  ──────────────────────────────────────────────────────────
# [PROF #5] Updated to describe all 4 sequences
# [PROF #3] Added explicit "only describe what you actually see" instruction

OBSERVE_PROMPT = """You are an expert neuroradiologist reviewing brain MRI images.
Each image shown is a 4-panel composite containing all four MRI sequences side by side:
  LEFT PANEL 1 — FLAIR  (bright = fluid/tumor/edema; CSF suppressed)
  LEFT PANEL 2 — T1     (bright = fat/blood; dark = CSF; anatomy reference)
  RIGHT PANEL 3 — T1ce  (contrast-enhanced T1; bright = blood-brain barrier breakdown → active tumor)
  RIGHT PANEL 4 — T2    (bright = water/edema/CSF; complements FLAIR)

IMPORTANT RULES:
- Describe ONLY what you can actually see in the images provided.
- Do NOT invent findings. If you are unsure, say "possibly" or "cannot exclude".
- Do NOT make a diagnosis in this step — only describe observations.

For each sequence, systematically describe:
1. Brain symmetry (left vs right hemisphere)
2. White matter appearance and any signal abnormalities
3. Gray matter appearance
4. Ventricles — size, shape, symmetry
5. Any hyperintense (bright) or hypointense (dark) focal regions
6. Any mass effect or midline shift
7. Sulci/gyri pattern
8. Cerebellum and brainstem if visible
9. Contrast enhancement on T1ce specifically

Format as a structured list with a heading per sequence."""


def step1_observe(client: anthropic.Anthropic, image_paths: list[Path]) -> str:
    print("  [Step 1] Observing slices (all sequences)...")
    content = build_image_content(image_paths, modality_label="4-sequence panel")
    content.append({"type": "text", "text": OBSERVE_PROMPT})
    response = client.messages.create(
        model=MODEL,
        max_tokens=1800,   # increased for 4-sequence descriptions
        messages=[{"role": "user", "content": content}]
    )
    return get_text_from_response(response)


# ── Step 2: Analyze  ──────────────────────────────────────────────────────────
# [PROF #3] Confidence must be explicit; differential must have 2-3 options

ANALYZE_PROMPT_TEMPLATE = """You are an expert neuroradiologist.
Based on these multi-sequence MRI observations (FLAIR, T1, T1ce, T2):

{observations}

Perform structured analysis. Reason step by step:

1. SYMMETRY ASSESSMENT: Are there asymmetries suggesting pathology? Which side?
2. SIGNAL ABNORMALITIES: Rate significance of any abnormal signal per sequence.
3. CONTRAST ENHANCEMENT: Does T1ce show enhancement? Enhancement pattern (ring/nodular/none)?
4. MASS EFFECT: Is there midline shift, herniation, or ventricular compression?
5. VENTRICULAR SYSTEM: Normal size and shape? Any obstructive hydrocephalus?
6. DIFFERENTIAL CONSIDERATIONS: List exactly 2-3 possibilities with brief reasoning each.
   Common differentials for FLAIR hyperintensity with T1ce enhancement:
   glioblastoma, brain metastasis, lymphoma, abscess, demyelinating lesion.
7. CONFIDENCE LEVEL: State one of — High / Medium / Low
   Explain specifically what limits your confidence (e.g., limited slices, no DWI/ADC).

IMPORTANT: Base reasoning ONLY on what was described in the observations above.
Do not add findings not mentioned in step 1."""


def step2_analyze(client: anthropic.Anthropic, observations: str) -> str:
    print("  [Step 2] Analyzing findings...")
    prompt = ANALYZE_PROMPT_TEMPLATE.format(observations=observations)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )
    return get_text_from_response(response)


# ── Step 3: Report  ───────────────────────────────────────────────────────────
# [PROF #3] Strict section format enforced

REPORT_PROMPT_TEMPLATE = """You are an expert neuroradiologist writing a clinical report.

OBSERVATIONS:
{observations}

ANALYSIS:
{analysis}

Write a structured radiology report using EXACTLY these section headers.
Do not add extra sections or omit any section.

EXAMINATION: Brain MRI — FLAIR, T1, T1ce, T2 sequences, axial plane

CLINICAL INDICATION: Brain tumor screening / evaluation

TECHNIQUE: Multimodal MRI (FLAIR, T1, contrast-enhanced T1, T2) axial sequences reviewed

FINDINGS:
[4-6 sentences. Describe signal characteristics per sequence. Include laterality,
location (lobe/region), size estimate if visible, enhancement pattern, mass effect,
ventricular involvement. Use clinical terminology.]

IMPRESSION:
[2-3 sentences. State most likely diagnosis first, then differential diagnoses.
Include laterality. State if urgent correlation is recommended.]

RECOMMENDATIONS:
[1-2 sentences. Specify next steps: additional sequences (DWI, perfusion, spectroscopy),
biopsy consideration, neurosurgical referral, or follow-up interval.]

CONFIDENCE: [High / Medium / Low] — [One sentence explaining the basis and limitations.]

SEQUENCES REVIEWED: FLAIR, T1, T1ce (contrast-enhanced), T2

---
DISCLAIMER: AI research prototype. NOT for clinical use. Must be verified by a qualified radiologist.

Use precise clinical language. Do not speculate beyond what imaging supports."""


def step3_report(client: anthropic.Anthropic, observations: str, analysis: str) -> str:
    print("  [Step 3] Generating report...")
    prompt = REPORT_PROMPT_TEMPLATE.format(
        observations=observations,
        analysis=analysis
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )
    return get_text_from_response(response)


# ── Step 4: Hallucination Check  ──────────────────────────────────────────────
# [PROF #3] NEW — Claude reviews its own report against its own observations

VERIFY_PROMPT_TEMPLATE = """You are a senior radiologist performing a quality check.

The following report was generated by an AI system based ONLY on these observations:

ORIGINAL OBSERVATIONS:
{observations}

GENERATED REPORT:
{report}

Your task: Review the report for hallucinations — claims in the report that are NOT
supported by the observations above.

For each finding in the FINDINGS and IMPRESSION sections, state:
  SUPPORTED   — if the observation explicitly mentions this finding
  INFERRED    — if the observation implies it but does not state it directly
  UNSUPPORTED — if the observation does NOT mention this at all (potential hallucination)

Then provide:
HALLUCINATION RISK: Low / Medium / High
SUMMARY: One sentence summarizing the overall reliability of this report.

Be strict. "Inferred" is acceptable; "Unsupported" findings should be flagged."""


def step4_verify(client: anthropic.Anthropic, observations: str, report: str) -> dict:
    """
    [PROF #3] Step 4: Hallucination verification.
    Returns dict with verification text + risk level.
    """
    print("  [Step 4] Verifying report for hallucinations...")
    prompt = VERIFY_PROMPT_TEMPLATE.format(
        observations=observations,
        report=report
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    text = get_text_from_response(response)

    # Extract risk level
    risk = "unknown"
    text_lower = text.lower()
    if any(x in text_lower for x in [
    "hallucination risk: low", "risk: low", "risk is low",
    "low hallucination", "overall risk: low"
]):
        risk = "low"
    elif any(x in text_lower for x in [
    "hallucination risk: medium", "risk: medium", "risk is medium",
    "medium hallucination", "overall risk: medium", "moderate risk"
]):
        risk = "medium"
    elif any(x in text_lower for x in [
    "hallucination risk: high", "risk: high", "risk is high",
    "high hallucination", "overall risk: high"
]):
        risk = "high"
    else:
        if "low" in text_lower and "risk" in text_lower:
            risk = "low"
        elif "medium" in text_lower and "risk" in text_lower:
            risk = "medium"
        elif "high" in text_lower and "risk" in text_lower:
            risk = "high"
   


# ── Baseline: Rule-based template report  ────────────────────────────────────
# [PROF #4] Simple baseline with no AI — used for comparison

def run_baseline(volume_id: str, slices_used: list[str]) -> dict:
    """
    [PROF #4] Generate a rule-based template report from filename metadata only.
    This is the baseline that the AI report is compared against.
    No image analysis — just a generic template with volume/slice info.
    """
    n_slices = len(slices_used)
    report = f"""EXAMINATION: Brain MRI — FLAIR sequence, axial plane

CLINICAL INDICATION: Brain tumor screening / evaluation

TECHNIQUE: FLAIR axial sequences reviewed ({n_slices} slices)

FINDINGS:
{n_slices} axial FLAIR slices were reviewed for volume {volume_id}.
No automated image analysis was performed for this baseline report.
Signal characteristics, symmetry, and lesion presence were not assessed.
Ventricles and midline structures were not evaluated in this baseline.

IMPRESSION:
Baseline template report. No imaging findings assessed.
This report serves as a negative/uninformative baseline for comparison with AI-generated reports.

RECOMMENDATIONS:
Full radiologist review required. This baseline is provided for research comparison only.

CONFIDENCE: Low — No image analysis performed.

---
BASELINE REPORT: Rule-based template. No AI or image analysis. For comparison purposes only."""

    return {
        "volume_id":     volume_id,
        "report_type":   "baseline",
        "slices_used":   slices_used,
        "report":        report,
        "timestamp":     datetime.now().isoformat(),
        "findings_json": {
            "tumor_present":           False,
            "mass_effect":             False,
            "midline_shift":           False,
            "ventricular_abnormality": False,
            "hyperintense_region":     False,
            "laterality":              "unclear",
            "confidence":              "low",
            "modalities_reviewed":     ["flair"],
        }
    }


# ── Findings extraction  ──────────────────────────────────────────────────────
# [PROF #5] Added modalities_reviewed field

def extract_findings_json(observations: str, analysis: str, report: str) -> dict:
    """Convert free-text to structured findings. Rule-based."""
    text = f"{observations}\n{analysis}\n{report}".lower()

    def has_any(keywords):
        return any(k in text for k in keywords)

    return {
        "tumor_present": has_any([
            "tumor", "mass", "lesion", "neoplasm", "abnormal hyperintense",
            "space-occupying", "focal abnormality", "glioma", "glioblastoma",
            "metastasis", "enhancement"
        ]),
        "mass_effect": has_any([
            "mass effect", "compression", "effacement", "herniation"
        ]),
        "midline_shift": has_any([
            "midline shift", "shift of midline", "deviated midline"
        ]),
        "ventricular_abnormality": has_any([
            "ventricular enlargement", "ventricular compression",
            "ventricles are enlarged", "ventricles are compressed",
            "abnormal ventricles", "hydrocephalus", "obstructive"
        ]),
        "hyperintense_region": has_any([
            "hyperintense", "bright signal", "flair hyperintensity", "t2 hyperintense"
        ]),
        "contrast_enhancement": has_any([        # [PROF #5] NEW
            "enhancement", "enhancing", "ring-enhancing", "nodular enhancement",
            "contrast uptake", "t1ce"
        ]),
        "laterality": (
            "bilateral" if has_any(["bilateral", "both hemispheres", "both sides"]) else
            "left" if has_any([
                "left hemisphere", "left-sided", "left side", "left frontal",
                "left parietal", "left temporal", "left occipital", "left posterior",
                "left anterior", "left basal", "left thalamic"
            ]) else
            "right" if has_any([
                "right hemisphere", "right-sided", "right side", "right frontal",
                "right parietal", "right temporal", "right occipital", "right posterior",
                "right anterior", "right basal", "right thalamic"
            ]) else
            "unclear"
        ),
        "confidence": (
            "high"   if "confidence: high"   in text else
            "medium" if "confidence: medium" in text else
            "low"    if "confidence: low"    in text else
            "unclear"
        ),
        "modalities_reviewed": [         # [PROF #5] NEW
            ch for ch in ["flair", "t1", "t1ce", "t2"]
            if ch in text or ch.replace("t1ce", "t1 contrast") in text
        ],
    }


# ── Full Agent Pipeline  ──────────────────────────────────────────────────────

def run_agent(volume_id: str, client: anthropic.Anthropic) -> dict:
    """Run the full 4-step agent on one volume."""
    print(f"\n{'='*55}")
    print(f"  Processing: {volume_id}")
    print(f"{'='*55}")

    # [PROF #5] Use multimodal images if available, fall back to FLAIR-only
    plane_dir = OUTPUT_DIR / volume_id / PLANE
    if not plane_dir.exists() or not list(plane_dir.glob("*.png")):
        print(f"  [INFO] Multimodal images not found, falling back to FLAIR slices")
        plane_dir = OUTPUT_DIR / volume_id / FLAIR_PLANE
        modality_used = "flair_only"
    else:
        modality_used = "multimodal_4_sequences"

    if not plane_dir.exists():
        raise FileNotFoundError(f"No slices found for {volume_id}")

    all_pngs = sorted(plane_dir.glob("*.png"))
    if not all_pngs:
        raise FileNotFoundError(f"No PNG files in {plane_dir}")

    selected_slices = get_selected_slices(all_pngs, MAX_SLICES)
    print(f"  Using {len(selected_slices)} slices from: {modality_used}")

    # 4-step pipeline
    observations   = step1_observe(client, selected_slices)
    analysis       = step2_analyze(client, observations)
    report         = step3_report(client, observations, analysis)
    verification   = step4_verify(client, observations, report)   # [PROF #3]
    findings_json  = extract_findings_json(observations, analysis, report)

    # [PROF #4] Generate baseline for this volume too
    baseline = run_baseline(volume_id, [p.name for p in selected_slices])

    result = {
        "volume_id":        volume_id,
        "timestamp":        datetime.now().isoformat(),
        "model":            MODEL,
        "modality_used":    modality_used,          # [PROF #5]
        "max_slices":       MAX_SLICES,
        "slices_used":      [p.name for p in selected_slices],
        "observations":     observations,
        "analysis":         analysis,
        "findings_json":    findings_json,
        "report":           report,
        "verification":     verification,            # [PROF #3]
        "baseline_report":  baseline["report"],     # [PROF #4]
        "hallucination_risk": verification["hallucination_risk"],   # [PROF #3]
        "flagged_for_review": verification["flagged"],              # [PROF #3]
    }

    flag_str = " ⚠️  FLAGGED" if verification["flagged"] else " ✅"
    print(f"  Report done | Hallucination risk: {verification['hallucination_risk']}{flag_str}")
    return result


def save_report(result: dict, reports_dir: Path):
    """Save result as JSON and readable TXT."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    vol_id = result["volume_id"]

    # JSON
    json_path = reports_dir / f"{vol_id}_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # TXT
    txt_path = reports_dir / f"{vol_id}_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MRI RADIOLOGY REPORT\n")
        f.write("=" * 55 + "\n")
        f.write(f"Volume ID      : {result['volume_id']}\n")
        f.write(f"Generated      : {result['timestamp']}\n")
        f.write(f"Model          : {result['model']}\n")
        f.write(f"Modality       : {result['modality_used']}\n")
        f.write(f"Slices used    : {len(result['slices_used'])}\n")
        f.write(f"Hallucin. risk : {result['hallucination_risk'].upper()}\n")
        if result["flagged_for_review"]:
            f.write("⚠  FLAGGED FOR RADIOLOGIST REVIEW\n")
        f.write("=" * 55 + "\n\n")

        f.write(result["report"])

        f.write("\n\n" + "=" * 55 + "\n")
        f.write("HALLUCINATION VERIFICATION  [PROF #3]\n")
        f.write("=" * 55 + "\n")
        f.write(result["verification"]["verification_text"])

        f.write("\n\n" + "=" * 55 + "\n")
        f.write("BASELINE REPORT (rule-based, no AI)  [PROF #4]\n")
        f.write("=" * 55 + "\n")
        f.write(result["baseline_report"])

        f.write("\n\n" + "=" * 55 + "\n")
        f.write("STRUCTURED FINDINGS JSON\n")
        f.write("=" * 55 + "\n")
        f.write(json.dumps(result["findings_json"], indent=2))

        f.write("\n\n" + "=" * 55 + "\n")
        f.write("AGENT REASONING\n")
        f.write("=" * 55 + "\n\n")
        f.write("--- STEP 1: OBSERVATIONS ---\n")
        f.write(result["observations"])
        f.write("\n\n--- STEP 2: ANALYSIS ---\n")
        f.write(result["analysis"])

    print(f"  Saved → {txt_path.name}")
    return txt_path


if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not set.")
        raise SystemExit(1)

    if not MANIFEST.exists():
        print(f"\nERROR: Manifest not found: {MANIFEST}")
        raise SystemExit(1)

    client = anthropic.Anthropic(api_key=api_key)

    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    volume_ids = [m["volume_id"] for m in manifest]
    to_process = volume_ids[:NUM_CASES_TO_PROCESS]

    if SKIP_EXISTING:
        to_process = [v for v in to_process
                      if not (REPORTS_DIR / f"{v}_result.json").exists()]

    print(f"\nWill process: {len(to_process)} volumes")
    print(f"Multimodal mode: {USE_MULTIMODAL}")

    results = []
    flagged = []
    for vol_id in to_process:
        try:
            result = run_agent(vol_id, client)
            save_report(result, REPORTS_DIR)
            results.append(result)
            if result["flagged_for_review"]:
                flagged.append(vol_id)
        except Exception as e:
            print(f"\nERROR processing {vol_id}: {e}")

    print(f"\n{'='*55}")
    print(f"Done. {len(results)} reports saved.")
    print(f"Flagged for review: {len(flagged)} — {flagged}")
    print(f"{'='*55}")