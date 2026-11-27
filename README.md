# MRI Report Generation Agent

AI-powered brain tumour radiology report generation using Claude (Anthropic) and the BraTS 2020 dataset.

## What it does
An end-to-end agentic AI pipeline that reads brain MRI scans and automatically generates structured radiology reports using a 4-step pipeline: **Observe → Analyze → Report → Verify**.

## Results
Evaluated on 106 BraTS 2020 volumes against an image-blind trivial baseline and a non-trivial FLAIR-threshold baseline, with 95% bootstrap confidence intervals.

| Metric | System | Trivial baseline | Threshold baseline |
|--------|--------|-------------------|---------------------|
| Tumor accuracy/precision/recall/F1 | 1.00 [1.00, 1.00] | 0.00 | 1.00 [1.00, 1.00] |
| Report quality (0-1) | 0.992 [0.980, 1.00] | 0.667 | - |
| Laterality, exact match | 0.293 [0.208, 0.377] | - | 0.745 [0.660, 0.821] |
| Laterality, soft match | 0.443 [0.368, 0.524] | - | 0.868 [0.826, 0.910] |

The accuracy/precision/recall/F1 numbers look perfect because every case in this sample has a tumor present (an all-positive set), so a baseline that always guesses "tumor" scores the same on those metrics. The laterality result is the more informative one: the system gets left/right wrong on the majority of unilateral cases, performing worse than the simple threshold baseline. See the paper for the full analysis, including a 9-case author-conducted structured review.

## Project Structure
- mri_pipeline.py — Data extraction and preprocessing
- mri_agent.py — 4-step Claude agentic pipeline
- evaluate_reports.py — Accuracy and quality evaluation
- app.py — Gradio web UI (localhost:7860)

## Dataset
BraTS 2020 (Brain Tumour Segmentation) — 369 patient volumes, HDF5 format.
Download from: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

## Setup
1. Install dependencies:
   pip install anthropic numpy pillow h5py tqdm matplotlib gradio rouge-score
2. Set API key:
   \ = "your-key-here"
3. Run in order:
   python mri_pipeline.py
   python mri_agent.py
   python evaluate_reports.py
   python app.py

## Tech Stack
- Claude claude-opus-4-5 (Anthropic) — Vision + Language model
- BraTS 2020 — Brain tumour MRI dataset
- Python — h5py, NumPy, Pillow, Gradio
- Gradio — Web UI

## Disclaimer
This is a research prototype. All outputs must be verified by a qualified radiologist before any clinical use.
