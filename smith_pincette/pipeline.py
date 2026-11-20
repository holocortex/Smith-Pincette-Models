"""Load a Smith-Pincette model and extract spans from text.

The decoding is the one the models were calibrated and evaluated with: per-class calibrated thresholds,
boundary snapping and length-weighted interval scheduling over 512-token windows (64-token overlap).
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from smith_pincette._core.decoding import _weighted_interval_scheduling, snap_span_boundaries
from smith_pincette._core.features import compute_lexical_features, extract_token_metadata

FILES = ["model.onnx", "config.json", "calibrated_thresholds.json", "tokenizer.json"]
# CUDA when available, else CPU; other providers (e.g. CoreML) are opt-in via providers=[...]
DEFAULT_PROVIDERS = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
LENGTH_WEIGHT = -0.35  # span priority = prob * (1 + LENGTH_WEIGHT * min(len, 30) / 30); slightly prefers shorter spans
# alphanumeric runs (Latin incl. accents, Greek, Cyrillic) stay whole; every other non-space character is its own word
_WORDS = re.compile(r"[A-Za-z0-9À-ɏͰ-ϿЀ-ӿ]+|[^\sA-Za-z0-9À-ɏͰ-ϿЀ-ӿ]")


@dataclass
class Span:
    start: int  # character offsets into the input text, end exclusive
    end: int
    label: str
    score: float
    text: str


class SmithPincette:
    def __init__(self, path: str, providers: Optional[List[str]] = None, window_batch_size: int = 8):
        self.tokenizer = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        self.tokenizer.no_truncation()  # long texts are split into windows below
        self.tokenizer.no_padding()
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        with open(os.path.join(path, "calibrated_thresholds.json"), encoding="utf-8") as f:
            self.thresholds: Dict[str, float] = json.load(f)
        self.max_tokens = cfg["max_tokens"]
        self.pad_id = cfg["pad_id"]
        self._labels: List[Tuple[str, str]] = [tuple(x) for x in cfg["labels"]]  # (label, canonical label) per query slot
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(os.path.join(path, "model.onnx"), opts, providers=providers or DEFAULT_PROVIDERS)
        self.window_batch_size = window_batch_size

    @property
    def labels(self) -> List[str]:
        """Every class the model can return."""
        return sorted({lbl for lbl, _ in self._labels if not lbl.startswith("query_")})

    @classmethod
    def from_pretrained(cls, name_or_path: str, revision: Optional[str] = None, **kwargs) -> "SmithPincette":
        """name_or_path: a Hugging Face repo id (e.g. "Holocortex/Smith-Pincette-Nitro") or a local directory with the model files.
        kwargs: providers (onnxruntime execution providers, default: CUDA if available, else CPU), window_batch_size."""
        if os.path.isdir(name_or_path):
            path = name_or_path
        else:
            from huggingface_hub import snapshot_download

            path = snapshot_download(name_or_path, revision=revision, allow_patterns=FILES)
        return cls(path, **kwargs)

    def _encode(self, text: str):
        words = [(m.start(), m.group()) for m in _WORDS.finditer(text)]
        enc = self.tokenizer.encode([w for _, w in words] or [""], is_pretokenized=True)
        # offsets of pre-split words are relative to the word
        offsets = [[words[i][0] + a, words[i][0] + e] if (i is not None and words and e > a) else [0, 0]
                   for i, (a, e) in zip(enc.word_ids, enc.offsets)]
        return np.asarray(enc.ids, dtype=np.int64), np.asarray(enc.attention_mask, dtype=np.int64), offsets

    def _cutoff(self, label: str, canonical: str, threshold: Optional[float]) -> float:
        if threshold is not None:
            return threshold
        return self.thresholds.get(canonical, self.thresholds.get(label, 0.5))

    def detect(self, text: str, threshold: Optional[float] = None, labels: Optional[Iterable[str]] = None) -> List[Span]:
        """Spans found in text, sorted by start. threshold: one flat cutoff for every class instead of the calibrated ones.
        labels: keep only these classes (e.g. {"email", "iban"})."""
        if not text.strip():
            return []
        ids_all, mask_all, offsets_all = self._encode(text)
        total = len(ids_all)
        if total <= self.max_tokens:
            windows = [(0, total)]
        else:
            overlap = min(64, self.max_tokens // 4)
            stride = max(self.max_tokens - overlap, self.max_tokens // 2)
            windows = []
            for s in range(0, total, stride):
                windows.append((s, min(s + self.max_tokens, total)))
                if windows[-1][1] == total:
                    break

        best: Dict[Tuple[int, int], Tuple[float, str]] = {}
        for c0 in range(0, len(windows), self.window_batch_size):
            chunk = windows[c0:c0 + self.window_batch_size]
            max_len = max(e - s for s, e in chunk)
            feeds = {k: [] for k in ("input_ids", "attention_mask", "regex_hints", "ortho_ids", "subword_cont_mask", "token_boundary_chars")}
            offs_per_win = []
            for s, e in chunk:
                offs = offsets_all[s:e]
                offs_per_win.append(offs)
                ortho, subword, chars = extract_token_metadata(text, offs, seq_len=e - s)
                for k, a, fill in (
                    ("input_ids", ids_all[s:e], self.pad_id),
                    ("attention_mask", mask_all[s:e], 0),
                    ("regex_hints", compute_lexical_features(text, offs), 0),
                    ("ortho_ids", ortho, 0),
                    ("subword_cont_mask", subword, 0),
                    ("token_boundary_chars", chars, 0),
                ):
                    pad = [(0, max_len - len(a))] + [(0, 0)] * (a.ndim - 1)
                    feeds[k].append(np.pad(a, pad, constant_values=fill))
            skip, probs_all, starts_all, ends_all = self.session.run(None, {k: np.stack(v) for k, v in feeds.items()})

            for w, offs in enumerate(offs_per_win):
                if skip[w]:  # the model's no-sensitive-data gate
                    continue
                probs, starts, ends = probs_all[w].astype(np.float64), starts_all[w], ends_all[w]  # probs: [queries, candidates]
                meta = self._labels[:probs.shape[0]]
                cut = np.array([self._cutoff(lbl, canon, threshold) for lbl, canon in meta], dtype=np.float64)
                for q, m in zip(*np.nonzero(probs >= cut[:, None])):
                    s_tok, e_tok = int(starts[m]), int(ends[m])
                    if s_tok >= len(offs) or e_tok >= len(offs):
                        continue
                    label = meta[q][0]
                    b_s, b_e = snap_span_boundaries(text, offs[s_tok][0], offs[e_tok][1], label)
                    p = float(probs_all[w][q, m])
                    if b_s < b_e and ((b_s, b_e) not in best or p > best[(b_s, b_e)][0]):
                        best[(b_s, b_e)] = (p, label)

        cands = [(s, e, lbl, p) for (s, e), (p, lbl) in best.items()]
        chosen = _weighted_interval_scheduling(cands, lambda c: c[3] * (1.0 + LENGTH_WEIGHT * min(max(1, c[1] - c[0]), 30) / 30.0))
        keep = set(labels) if labels is not None else None
        return [Span(s, e, lbl, p, text[s:e]) for s, e, lbl, p in sorted(chosen) if keep is None or lbl in keep]

    def mask(self, text: str, threshold: Optional[float] = None, labels: Optional[Iterable[str]] = None, placeholder: str = "[{label}]") -> str:
        """text with every detected span replaced by placeholder (formatted with label=SPAN_LABEL_UPPERCASE)."""
        out, last = [], 0
        for sp in self.detect(text, threshold=threshold, labels=labels):
            out += [text[last:sp.start], placeholder.format(label=sp.label.upper())]
            last = sp.end
        return "".join(out) + text[last:]
