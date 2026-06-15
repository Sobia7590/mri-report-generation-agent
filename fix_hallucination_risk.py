import json
from pathlib import Path

REPORTS_DIR = Path(r"D:\Downloads\Agent\output\reports")

for json_path in REPORTS_DIR.glob("*_result.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    # Get verification text safely — works even if key is missing
    verification = result.get("verification", {})
    verify_text  = verification.get("verification_text", "").lower()

    # Also scan the full report text as fallback
    full_text = (
        result.get("observations", "") + " " +
        result.get("analysis", "") + " " +
        result.get("report", "") + " " +
        verify_text
    ).lower()

    risk = "unknown"
    if "risk" in full_text:
        if "low" in full_text:
            risk = "low"
        elif "medium" in full_text or "moderate" in full_text:
            risk = "medium"
        elif "high" in full_text:
            risk = "high"

    # Update safely whether verification key exists or not
    result["hallucination_risk"]  = risk
    result["flagged_for_review"]  = risk in ("medium", "high")

    if isinstance(verification, dict):
        verification["hallucination_risk"] = risk
        result["verification"] = verification
    else:
        result["verification"] = {"hallucination_risk": risk, "verification_text": ""}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"{json_path.stem}: → {risk}")

print("\nDone!")