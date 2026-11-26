import os
import json
from pathlib import Path
import anthropic

from mri_agent import step4_verify, REPORTS_DIR

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return
    client = anthropic.Anthropic(api_key=api_key)

    result_files = sorted(REPORTS_DIR.glob("*_result.json"))
    print(f"Re-verifying {len(result_files)} cases...")

    for json_path in result_files:
        with open(json_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        observations = result.get("observations", "")
        report = result.get("report", "")
        if not observations or not report:
            print(f"  SKIP {json_path.stem}: missing observations/report")
            continue

        verification = step4_verify(client, observations, report)

        result["verification"] = verification
        result["hallucination_risk"] = verification["hallucination_risk"]
        result["flagged_for_review"] = verification["flagged"]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  {json_path.stem}: -> {verification['hallucination_risk']}")

    print("\nDone! Now run evaluate_reports.py to refresh the summary.")

if __name__ == "__main__":
    main()