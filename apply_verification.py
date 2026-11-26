import json, sys
from pathlib import Path

REPORTS_DIR = Path(r"D:\Downloads\Agent\output\reports")

def parse_risk(text):
    t = text.lower()
    if any(x in t for x in ["hallucination risk: low", "risk: low", "risk is low", "low hallucination", "overall risk: low"]):
        return "low"
    if any(x in t for x in ["hallucination risk: medium", "risk: medium", "risk is medium", "medium hallucination", "overall risk: medium", "moderate risk"]):
        return "medium"
    if any(x in t for x in ["hallucination risk: high", "risk: high", "risk is high", "high hallucination", "overall risk: high"]):
        return "high"
    if "risk" in t:
        if "low" in t: return "low"
        if "medium" in t: return "medium"
        if "high" in t: return "high"
    return "unknown"

vol_id, response_path = sys.argv[1], sys.argv[2]
json_path = REPORTS_DIR / f"{vol_id}_result.json"

with open(json_path, "r", encoding="utf-8") as f:
    result = json.load(f)
with open(response_path, "r", encoding="utf-8") as f:
    text = f.read()

risk = parse_risk(text)
result["verification"] = {"verification_text": text, "hallucination_risk": risk, "flagged": risk in ("medium", "high")}
result["hallucination_risk"] = risk
result["flagged_for_review"] = risk in ("medium", "high")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"{vol_id}: -> {risk}")