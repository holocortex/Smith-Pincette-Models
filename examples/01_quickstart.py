"""Download a Smith-Pincette model from Hugging Face and print the spans it finds.

    python examples/01_quickstart.py
    python examples/01_quickstart.py --model Holocortex/Smith-Pincette-Sprint
"""

import argparse

from smith_pincette import SmithPincette

TEXT = (
    "Hi team, please refund Maria Schneider (maria.schneider@example.com, +49 170 1234567). "
    "IBAN DE89 3704 0044 0532 0130 00. Ship to Königsstraße 12, 10115 Berlin."
)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Holocortex/Smith-Pincette-Nitro")
ap.add_argument("--revision", default="v0.5-preview")
args = ap.parse_args()

model = SmithPincette.from_pretrained(args.model, revision=args.revision)
for span in model.detect(TEXT):
    print(f"{span.label:<20} {span.score:.2f}  {span.text!r}")
print()
print(model.mask(TEXT))
