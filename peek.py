import json
from pathlib import Path

REPORTS_DIR = Path(r"D:\Downloads\Agent\output\reports")

volumes = ["volume_100", "volume_103", "volume_112", "volume_101", "volume_108"]

for vol in volumes:
    path = REPORTS_DIR / f"{vol}_result.json"
    if not path.exists():
        print(f"\n===== {vol}: FILE NOT FOUND =====")
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n===== {vol} =====")
    print("--- REPORT ---")
    print(data.get("report", "(none)"))
    print("\n--- VERIFICATION TEXT ---")
    print(data.get("verification", {}).get("verification_text", "(none)"))
    print("-" * 60)