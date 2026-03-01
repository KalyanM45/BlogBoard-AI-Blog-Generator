#!/usr/bin/env python3
"""
build_schedule.py
Reads all input JSONs from backend/data/input/ and merges them into backend/schedule.json.
"""

import json
import os
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────
BACKEND_DIR  = Path(__file__).parent.parent          # BlogBoard/backend/
INPUT_DIR    = BACKEND_DIR / "data" / "input"
OUTPUT_FILE  = BACKEND_DIR / "schedule.json"

# ── Filename → domain mapping ────────────────────────────
DOMAIN_MAP = {
    "machine_learning.json":              "ml",
    "deep_learning.json":                 "dl",
    "natural_language_processing.json":   "nlp",
    "computer_vision.json":               "cv",
    "generative_ai.json":                 "genai",
    "statistics.json":                    "statistics",
}

def build_schedule():
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    schedule = {}
    total = 0
    conflicts = []

    for filename, domain in DOMAIN_MAP.items():
        filepath = INPUT_DIR / filename
        if not filepath.exists():
            print(f"[WARN]  Skipping missing file: {filename}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            date  = entry.get("date", "").strip()
            topic = entry.get("topic", "").strip()

            if not date or not topic:
                print(f"[WARN]  Skipping entry with missing date/topic in {filename}: {entry}")
                continue

            if date in schedule:
                conflicts.append((date, schedule[date]["domain"], domain))
                print(f"[WARN]  Date conflict: {date} already assigned to {schedule[date]['domain']}, overwriting with {domain}")

            schedule[date] = {"domain": domain, "topic": topic}
            total += 1

    # Sort by date
    schedule = dict(sorted(schedule.items()))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)

    print(f"\n✅ schedule.json built successfully.")
    print(f"   Total entries : {total}")
    print(f"   Total dates   : {len(schedule)}")
    print(f"   Conflicts     : {len(conflicts)}")
    print(f"   Output path   : {OUTPUT_FILE}")

if __name__ == "__main__":
    build_schedule()
