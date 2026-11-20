import os

import pytest

from smith_pincette import SmithPincette, Span
from smith_pincette._core.decoding import _weighted_interval_scheduling


def test_interval_scheduling_keeps_best_non_overlapping_set():
    cands = [(0, 10, "a", 0.6), (0, 4, "b", 0.5), (5, 10, "c", 0.5), (12, 15, "d", 0.9)]
    chosen = sorted(_weighted_interval_scheduling(cands, lambda c: c[3]))
    assert chosen == [(0, 4, "b", 0.5), (5, 10, "c", 0.5), (12, 15, "d", 0.9)]


def test_mask_replaces_spans_in_order():
    sp = SmithPincette.__new__(SmithPincette)
    text = "mail bob@x.org or call 123"
    sp.detect = lambda t, threshold=None, labels=None: [Span(5, 14, "email", 0.9, "bob@x.org"), Span(23, 26, "phone_number", 0.8, "123")]
    assert sp.mask(text) == "mail [EMAIL] or call [PHONE_NUMBER]"
    assert sp.mask(text, placeholder="<{label}>") == "mail <EMAIL> or call <PHONE_NUMBER>"


@pytest.mark.skipif(not os.getenv("SMITH_PINCETTE_MODEL"), reason="set SMITH_PINCETTE_MODEL to a repo id or local model dir")
def test_real_model_finds_email_and_iban():
    model = SmithPincette.from_pretrained(os.environ["SMITH_PINCETTE_MODEL"])
    text = "Please refund Maria Schneider (maria.schneider@example.com), IBAN DE89 3704 0044 0532 0130 00."
    found = {(s.label, s.text) for s in model.detect(text)}
    assert ("email", "maria.schneider@example.com") in found
    assert ("iban", "DE89 3704 0044 0532 0130 00") in found
    assert model.detect("   ") == []
