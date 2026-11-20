"""List the classes, keep only some of them, and trade recall for precision with a flat threshold.

By default every class uses its own calibrated threshold. A flat threshold replaces all of them:
lower finds more (and more false positives), higher finds less.
"""

from smith_pincette import SmithPincette

model = SmithPincette.from_pretrained("Holocortex/Smith-Pincette-Nitro", revision="v0.5-preview")
print("classes:", ", ".join(model.labels))

text = "Ticket #4471 from Jane Doe (j.doe@example.org, +44 20 7946 0958): please update my address to 221B Baker Street, London."

print("\ncalibrated thresholds:")
for s in model.detect(text):
    print(f"  {s.label:<22} {s.score:.2f} {s.text!r}")

print("\nonly contact details:")
for s in model.detect(text, labels={"email", "phone_number"}):
    print(f"  {s.label:<22} {s.score:.2f} {s.text!r}")

for threshold in (0.3, 0.7):
    found = model.detect(text, threshold=threshold)
    print(f"\nflat threshold {threshold}: {[(s.label, s.text) for s in found]}")
