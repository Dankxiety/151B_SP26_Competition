import json
import csv

INPUT_FILE = "priv_results_final.jsonl"
OUTPUT_FILE = "submission.csv"

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as fout:

    writer = csv.writer(
        fout,
        quoting=csv.QUOTE_ALL,  # safely handles commas, quotes, newlines
    )

    writer.writerow(["id", "response"])

    for line in fin:
        line = line.strip()
        if not line:
            continue

        record = json.loads(line)

        writer.writerow([
            record["id"],
            record["response"],
        ])

print(f"Wrote {OUTPUT_FILE}")