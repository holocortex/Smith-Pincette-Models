"""Run a model over a JSONL file and write the spans of every record.

Input:  one JSON object per line with a "text" field (other fields are kept).
Output: the same records plus "spans" and "masked".

    python examples/03_batch_jsonl.py input.jsonl output.jsonl
"""

import argparse
import json
import time
from dataclasses import asdict

from smith_pincette import SmithPincette

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("output")
ap.add_argument("--model", default="Holocortex/Smith-Pincette-Nitro")
ap.add_argument("--revision", default="v0.5-preview")
args = ap.parse_args()

model = SmithPincette.from_pretrained(args.model, revision=args.revision)
n, t0 = 0, time.perf_counter()
with open(args.input, encoding="utf-8") as src, open(args.output, "w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        record = json.loads(line)
        spans = model.detect(record["text"])
        record["spans"] = [asdict(s) for s in spans]
        record["masked"] = model.mask(record["text"])
        dst.write(json.dumps(record, ensure_ascii=False) + "\n")
        n += 1
print(f"{n} records in {time.perf_counter() - t0:.1f}s -> {args.output}")
