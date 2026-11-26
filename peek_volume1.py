import json
from pathlib import Path

REPORTS_DIR = Path(r"D:\Downloads\Agent\output\reports")
with open(REPORTS_DIR / "volume_1_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("--- MODALITY USED ---")
print(data.get("modality_used"))
print("\n--- SLICES USED ---")
print(data.get("slices_used"))
print("\n--- OBSERVATIONS ---")
print(data.get("observations", "(none)"))
print("\n--- ANALYSIS ---")
print(data.get("analysis", "(none)"))
print("\n--- REPORT ---")
print(data.get("report", "(none)"))
print("\n--- FINDINGS JSON ---")
print(json.dumps(data.get("findings_json", {}), indent=2))