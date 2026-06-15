"""
Gradio Demo App — IMPROVED
===========================
Changes addressing professor feedback:

  [PROF #1] Shows actual evaluation numbers (accuracy, F1, quality) in UI
  [PROF #2] Expert radiologist scoring panel — fill scores in-app
  [PROF #3] Hallucination risk badge + flagged warning shown prominently
  [PROF #4] Side-by-side AI report vs Baseline report tab
  [PROF #5] Multimodal gallery (shows all 4 sequences per slice)
  [PROF #6] Overlay gallery (GT tumor mask overlaid on FLAIR)
"""

import json
from pathlib import Path
import gradio as gr

OUTPUT_DIR  = Path(r"D:\Downloads\Agent\output\slices")
REPORTS_DIR = Path(r"D:\Downloads\Agent\output\reports")

EVAL_CASE_TABLE  = REPORTS_DIR / "evaluation_case_table.json"
EVAL_SUMMARY     = REPORTS_DIR / "evaluation_summary.json"
EXPERT_SCORES_PATH = REPORTS_DIR / "expert_scores.json"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_volume_ids():
    if not REPORTS_DIR.exists():
        return []
    ids = [p.stem.replace("_result", "") for p in REPORTS_DIR.glob("*_result.json")]
    try:
        return sorted(ids, key=lambda v: int(v.split("_")[-1]))
    except Exception:
        return sorted(ids)

def load_eval_lookup():
    if not EVAL_CASE_TABLE.exists():
        return {}
    with open(EVAL_CASE_TABLE, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {r["volume_id"]: r for r in rows}

def load_eval_summary():
    if not EVAL_SUMMARY.exists():
        return {}
    with open(EVAL_SUMMARY, "r", encoding="utf-8") as f:
        return json.load(f)

def load_expert_scores():
    if not EXPERT_SCORES_PATH.exists():
        return {}
    with open(EXPERT_SCORES_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {r["volume_id"]: r for r in rows}

EVAL_LOOKUP    = load_eval_lookup()
EVAL_SUMMARY_D = load_eval_summary()
EXPERT_LOOKUP  = load_expert_scores()


# ── [PROF #1] Aggregate metrics panel ────────────────────────────────────────

def build_aggregate_summary() -> str:
    """[PROF #1] Show real evaluation numbers."""
    if not EVAL_SUMMARY_D:
        return "No evaluation_summary.json found yet.\nRun evaluate_reports.py first."

    ai = EVAL_SUMMARY_D.get("ai_metrics", {})
    bl = EVAL_SUMMARY_D.get("baseline_metrics", {})
    n  = EVAL_SUMMARY_D.get("n_cases", 0)

    lines = [
        f"{'='*45}",
        f"EVALUATION RESULTS  ({n} cases)  [PROF #1]",
        f"{'='*45}",
        f"",
        f"TUMOR DETECTION",
        f"  Accuracy  : {ai.get('accuracy','?')}   (Baseline: {bl.get('accuracy','?')})",
        f"  Precision : {ai.get('precision','?')}   (Baseline: {bl.get('precision','?')})",
        f"  Recall    : {ai.get('recall','?')}   (Baseline: {bl.get('recall','?')})",
        f"  F1 Score  : {ai.get('f1','?')}   (Baseline: {bl.get('f1','?')})",
        f"",
        f"LATERALITY",
        f"  Exact accuracy : {ai.get('laterality_accuracy','?')}",
        f"  Soft score     : {ai.get('avg_laterality_soft','?')}",
        f"",
        f"REPORT QUALITY",
        f"  Avg quality (AI)       : {ai.get('avg_quality_score','?')}",
        f"  Avg quality (Baseline) : {bl.get('avg_quality_score','?')}",
        f"",
        f"HALLUCINATION  [PROF #3]",
        f"  % Flagged for review : {ai.get('pct_flagged','?')}%",
        f"  Risk breakdown       : {ai.get('hallucination_breakdown',{})}",
        f"",
        f"MULTIMODAL  [PROF #5]",
        f"  Cases using all 4 sequences : {ai.get('multimodal_cases','?')}",
    ]

    expert = ai.get("expert_summary", {})
    if expert:
        lines += [
            f"",
            f"EXPERT VALIDATION  [PROF #2]",
            f"  Cases reviewed          : {expert.get('n_expert_reviewed','?')}",
            f"  Mean accuracy (1-5)     : {expert.get('mean_accuracy','?')}",
            f"  Mean completeness (1-5) : {expert.get('mean_completeness','?')}",
            f"  Mean usefulness (1-5)   : {expert.get('mean_clinical_usefulness','?')}",
            f"  Would use as draft      : {expert.get('pct_would_use_as_draft','?')}%",
        ]

    return "\n".join(lines)


# ── Per-case loader ────────────────────────────────────────────────────────────

def load_case(volume_id):
    """Returns: ai_gallery, overlay_gallery, multimodal_gallery,
                report_text, baseline_text, summary_text, risk_badge"""
    if not volume_id:
        return [], [], [], "", "", "", "No case selected"

    json_path = REPORTS_DIR / f"{volume_id}_result.json"
    if not json_path.exists():
        return [], [], [], "No report found.", "", "No summary.", "N/A"

    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    report_text   = result.get("report", "No report.")
    baseline_text = result.get("baseline_report", "No baseline report.")
    slices_used   = result.get("slices_used", [])
    modality      = result.get("modality_used", "flair_only")
    risk          = result.get("hallucination_risk", "unknown")
    flagged       = result.get("flagged_for_review", False)

    # ── [PROF #5] Multimodal gallery ──────────────────────────────────────────
    multi_dir   = OUTPUT_DIR / volume_id / "multimodal"
    flair_dir   = OUTPUT_DIR / volume_id / "slices"
    overlay_dir = OUTPUT_DIR / volume_id / "overlays"    # [PROF #6]

    ai_gallery        = []
    multimodal_gallery = []
    overlay_gallery   = []

    # AFTER
    for slice_name in slices_used:
        import re
        m = re.search(r"slice[_\-](\d+)", slice_name)
        num = m.group(1) if m else slice_name.replace(".png", "")

    # FLAIR gallery — try both naming formats
    for flair_name in [f"flair_slice_{num}.png", slice_name]:
        flair_path = flair_dir / flair_name
        if flair_path.exists():
            ai_gallery.append((str(flair_path), f"FLAIR slice {num}"))
            break

    # Multimodal gallery
    multi_path = multi_dir / f"multimodal_slice_{num}.png"
    if multi_path.exists():
        multimodal_gallery.append((str(multi_path), f"4-seq slice {num}"))

    # Overlay gallery
    overlay_path = overlay_dir / f"overlay_slice_{num}.png"
    if overlay_path.exists():
        overlay_gallery.append((str(overlay_path), f"GT overlay slice {num}"))

    # ── Evaluation summary text ────────────────────────────────────────────────
    eval_row = EVAL_LOOKUP.get(volume_id, {})
    expert   = EXPERT_LOOKUP.get(volume_id, {})

    lines = [
        f"Volume: {volume_id}",
        f"Modality: {modality}  [PROF #5]",
        f"Slices reviewed: {len(slices_used)}",
        "",
        "── GROUND TRUTH vs AI PREDICTION ──",
    ]

    if eval_row:
        lines += [
            f"GT Tumor:      {eval_row.get('gt_tumor_present')}",
            f"AI Tumor:      {eval_row.get('pred_tumor_present')}",
            f"Match:         {'✅' if eval_row.get('tumor_presence_match') else '❌'}",
            f"",
            f"GT Laterality: {eval_row.get('gt_laterality')}",
            f"AI Laterality: {eval_row.get('pred_laterality')}",
            f"Lat Match:     {'✅' if eval_row.get('laterality_match') else '❌'}",
            f"",
            f"GT Burden:     {eval_row.get('gt_tumor_burden')}",
            f"AI Burden:     {eval_row.get('pred_burden')}",
            f"Quality Score: {eval_row.get('quality_score')}",
            f"",
            f"── HALLUCINATION  [PROF #3] ──",
            f"Risk:          {risk.upper()}",
            f"Flagged:       {'⚠ YES — needs radiologist review' if flagged else '✅ No'}",
        ]

    if expert:                           # [PROF #2]
        lines += [
            "",
            "── EXPERT RADIOLOGIST SCORE  [PROF #2] ──",
            f"Accuracy (1-5):    {expert.get('accuracy', 'pending')}",
            f"Completeness:      {expert.get('completeness', 'pending')}",
            f"Usefulness:        {expert.get('clinical_usefulness', 'pending')}",
            f"Would use draft:   {expert.get('would_use_as_draft', 'pending')}",
            f"Notes: {expert.get('notes', '')}",
        ]
    else:
        lines += [
            "",
            "── EXPERT SCORE  [PROF #2] ──",
            "Not yet reviewed by expert.",
        ]

    # [PROF #3] Risk badge string
    badge = {
        "low":     "🟢 Hallucination Risk: LOW",
        "medium":  "🟡 Hallucination Risk: MEDIUM  ⚠",
        "high":    "🔴 Hallucination Risk: HIGH  ⚠⚠  NEEDS REVIEW",
        "unknown": "⚪ Hallucination Risk: UNKNOWN",
    }.get(risk, "⚪ Unknown")

    return (
        ai_gallery,
        overlay_gallery,
        multimodal_gallery,
        report_text,
        baseline_text,
        "\n".join(lines),
        badge
    )


# ── [PROF #2] Save expert score ───────────────────────────────────────────────

def save_expert_score(volume_id, accuracy, completeness, usefulness, would_use, notes):
    """Save radiologist rating from the UI."""
    if not volume_id:
        return "No volume selected."
    scores = list(EXPERT_LOOKUP.values())
    new_entry = {
        "volume_id":           volume_id,
        "accuracy":            int(accuracy),
        "completeness":        int(completeness),
        "clinical_usefulness": int(usefulness),
        "would_use_as_draft":  bool(would_use),
        "notes":               notes,
    }
    # Update or add
    existing = [s for s in scores if s["volume_id"] != volume_id]
    existing.append(new_entry)
    EXPERT_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERT_SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    EXPERT_LOOKUP[volume_id] = new_entry
    return f"✅ Score saved for {volume_id}"


# ── Build UI ──────────────────────────────────────────────────────────────────

volume_choices = load_volume_ids()

with gr.Blocks(title="MRI Report Agent Demo") as demo:
    gr.Markdown("# 🧠 MRI Report Generation Agent — Final Demo")
    gr.Markdown("*Research prototype — all reports must be verified by a qualified radiologist*")

    with gr.Row():
        volume_dropdown = gr.Dropdown(
            choices=volume_choices,
            value=volume_choices[0] if volume_choices else None,
            label="Select Volume",
            scale=3
        )
        load_btn = gr.Button("Load Case", variant="primary", scale=1)

    # [PROF #3] Hallucination risk badge at top
    risk_badge = gr.Markdown("Select a case to see hallucination risk")

    with gr.Tabs():

        # ── Tab 1: MRI Viewer ─────────────────────────────────────────────────
        with gr.Tab("📷 MRI Slices"):
            gr.Markdown("### FLAIR Slices used by Agent")
            ai_gallery = gr.Gallery(label="FLAIR slices", columns=5, height=220)

            gr.Markdown("### All 4 Sequences (FLAIR · T1 · T1ce · T2)  [PROF #5]")
            multi_gallery = gr.Gallery(label="Multimodal panels", columns=3, height=280)

            gr.Markdown("### GT Tumor Mask Overlays  [PROF #6]  *(red=necrotic, yellow=edema, green=enhancing)*")
            overlay_gallery = gr.Gallery(label="Overlay images", columns=5, height=220)

        # ── Tab 2: AI Report ──────────────────────────────────────────────────
        with gr.Tab("📋 AI Report"):
            report_box = gr.Textbox(label="Generated Radiology Report", lines=28)

        # ── Tab 3: Baseline Comparison  [PROF #4] ─────────────────────────────
        with gr.Tab("⚖️ Baseline Comparison  [PROF #4]"):
            gr.Markdown("AI report (left) vs Rule-based baseline with no AI (right)")
            with gr.Row():
                ai_report_cmp  = gr.Textbox(label="AI-Generated Report", lines=22)
                bl_report_cmp  = gr.Textbox(label="Baseline (No AI)", lines=22)

        # ── Tab 4: Evaluation Summary  [PROF #1] ─────────────────────────────
        with gr.Tab("📊 Evaluation  [PROF #1]"):
            with gr.Row():
                case_summary   = gr.Textbox(label="This Case", lines=28, scale=1)
                agg_summary    = gr.Textbox(
                    label="Aggregate Metrics (all cases)",
                    value=build_aggregate_summary(),
                    lines=28, scale=1
                )

        # ── Tab 5: Expert Scoring  [PROF #2] ─────────────────────────────────
        with gr.Tab("👩‍⚕️ Expert Scoring  [PROF #2]"):
            gr.Markdown("### Radiologist / Expert Review Panel")
            gr.Markdown("Rate the AI-generated report for the selected volume. "
                        "Scores are saved to `expert_scores.json` and included in evaluation.")
            with gr.Row():
                acc_slider  = gr.Slider(1, 5, step=1, value=3, label="Accuracy (1=wrong, 5=excellent)")
                comp_slider = gr.Slider(1, 5, step=1, value=3, label="Completeness")
                use_slider  = gr.Slider(1, 5, step=1, value=3, label="Clinical Usefulness")
            would_use    = gr.Checkbox(label="Would use this report as a draft?", value=False)
            expert_notes = gr.Textbox(label="Notes / Comments", lines=4)
            save_btn     = gr.Button("💾 Save Expert Score", variant="primary")
            save_status  = gr.Markdown("")

    # ── Wire up events ─────────────────────────────────────────────────────────

    def on_load(vol_id):
        gallery, overlays, multi, report, baseline, summary, badge = load_case(vol_id)
        return gallery, overlays, multi, report, report, baseline, summary, f"**{badge}**"

    load_btn.click(
        fn=on_load,
        inputs=volume_dropdown,
        outputs=[ai_gallery, overlay_gallery, multi_gallery,
                 report_box, ai_report_cmp, bl_report_cmp,
                 case_summary, risk_badge]
    )
    volume_dropdown.change(
        fn=on_load,
        inputs=volume_dropdown,
        outputs=[ai_gallery, overlay_gallery, multi_gallery,
                 report_box, ai_report_cmp, bl_report_cmp,
                 case_summary, risk_badge]
    )
    if volume_choices:
        demo.load(
            fn=on_load,
            inputs=volume_dropdown,
            outputs=[ai_gallery, overlay_gallery, multi_gallery,
                     report_box, ai_report_cmp, bl_report_cmp,
                     case_summary, risk_badge]
        )

    save_btn.click(
        fn=save_expert_score,
        inputs=[volume_dropdown, acc_slider, comp_slider,
                use_slider, would_use, expert_notes],
        outputs=save_status
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True
    )