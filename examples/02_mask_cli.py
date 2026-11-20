"""Mask a text file (or stdin) and write the result to stdout.

    python examples/02_mask_cli.py notes.txt > notes.masked.txt
    cat app.log | python examples/02_mask_cli.py --labels email,phone_number,person
"""

import argparse
import sys

from smith_pincette import SmithPincette

ap = argparse.ArgumentParser()
ap.add_argument("file", nargs="?", help="input file; stdin when omitted")
ap.add_argument("--model", default="Holocortex/Smith-Pincette-Nitro")
ap.add_argument("--revision", default="v0.5-preview")
ap.add_argument("--labels", help="comma-separated classes to mask; all classes when omitted")
ap.add_argument("--placeholder", default="[{label}]", help="replacement, {label} is the upper-case class name")
args = ap.parse_args()

text = open(args.file, encoding="utf-8").read() if args.file else sys.stdin.read()
model = SmithPincette.from_pretrained(args.model, revision=args.revision)
labels = args.labels.split(",") if args.labels else None
sys.stdout.write(model.mask(text, labels=labels, placeholder=args.placeholder))
