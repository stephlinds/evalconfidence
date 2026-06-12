"""Export generation logs to the committed scores-only CSV.

The raw `.eval` logs embed every GPQA question verbatim and therefore stay
local (logs/ is gitignored; the gated dataset's terms ask that questions not
be reposted). This script strips them down to ItemResult columns — item_id is
the opaque HF record id, the scores are our own — which is everything the demo
notebook needs and nothing the dataset license restricts.

Run after a generation run:  python examples/export_scores.py
"""

import csv
from pathlib import Path

from evalconfidence import from_inspect

LOG_DIR = Path(__file__).parent.parent / "logs" / "full"
OUT = Path(__file__).parent / "data" / "gpqa_diamond_scores.csv"


def main():
    logs = sorted(LOG_DIR.glob("*.eval"))
    if not logs:
        raise SystemExit(f"No .eval logs in {LOG_DIR}; run the generation first.")
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "model_id", "score", "epoch"])
        for log in logs:
            results = from_inspect(str(log))
            writer.writerows(
                (r.item_id, r.model_id, r.score, r.epoch) for r in results
            )
            print(f"{log.name}: {len(results)} rows ({results[0].model_id})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
