# Smith-Pincette Models

Python examples for running the **Smith-Pincette Model Family** locally. The models find personal data in text (names,
contact details, addresses, ID and bank numbers, and more) and return exact character spans, so you can mask it before
text reaches logs, LLMs or third parties. Everything runs on your machine, on CPU or GPU; no text is sent anywhere.

**This is a preview release (v0.5-preview)**: expect rough edges, and expect the API and models to change.

| Model | Parameters | Use it for |
| :--- | :--- | :--- |
| [Smith-Pincette-Nitro](https://huggingface.co/Holocortex/Smith-Pincette-Nitro) v0.5-preview | 146.7M | the fast default: CPU, high volume |
| [Smith-Pincette-Sprint](https://huggingface.co/Holocortex/Smith-Pincette-Sprint) v0.5-preview | 321.9M | the larger encoder, same 43 classes |

## Install

Python 3.10 or newer.

```bash
pip install "git+https://github.com/holocortex/Smith-Pincette-Models"
# or, to run the examples from a checkout:
git clone https://github.com/holocortex/Smith-Pincette-Models && cd Smith-Pincette-Models
pip install -e .
```

The models are ONNX graphs and run with [onnxruntime](https://onnxruntime.ai); PyTorch is not needed. The first call
downloads the model from Hugging Face (Nitro ~0.6 GB, Sprint ~1.3 GB) into the usual Hugging Face cache.
For NVIDIA GPUs, install `onnxruntime-gpu` instead of `onnxruntime`; the GPU is then used automatically.

## Use

```python
from smith_pincette import SmithPincette

model = SmithPincette.from_pretrained("Holocortex/Smith-Pincette-Nitro", revision="v0.5-preview")

text = "Please refund Maria Schneider (maria.schneider@example.com), IBAN DE89 3704 0044 0532 0130 00."
for span in model.detect(text):
    print(span.label, round(span.score, 2), span.text)
# person 1.0 Maria Schneider
# email 1.0 maria.schneider@example.com
# iban 1.0 DE89 3704 0044 0532 0130 00

print(model.mask(text))
# Please refund [PERSON] ([EMAIL]), IBAN [IBAN].
```

- `detect(text, labels={"email", "iban"})`: keep only some classes; `model.labels` lists all 43.
- `detect(text, threshold=0.4)`: one flat cutoff instead of the calibrated per-class thresholds.
- `mask(text, placeholder="<{label}>")`: choose the replacement.
- `from_pretrained(path)` also takes a local directory with the model files (offline use).
- Pin `revision="v0.5-preview"` to stay on this release when newer versions are published.

## Examples

| Script | What it shows |
| :--- | :--- |
| [`01_quickstart.py`](examples/01_quickstart.py) | download a model, print spans and the masked text (`--model` to pick Nitro or Sprint) |
| [`02_mask_cli.py`](examples/02_mask_cli.py) | mask a file or stdin from the command line |
| [`03_batch_jsonl.py`](examples/03_batch_jsonl.py) | run over a JSONL file, write spans and masked text per record |
| [`04_labels_and_thresholds.py`](examples/04_labels_and_thresholds.py) | list classes, filter classes, flat thresholds |
| [`05_reversible_masking.py`](examples/05_reversible_masking.py) | replace values with numbered placeholders before an LLM call and restore them in the answer |
| [`06_mcp_server.py`](examples/06_mcp_server.py) | local MCP server so coding agents can detect and mask without sending data out (`pip install ".[mcp]"`) |

## Limitations

The models are statistical: they miss some sensitive values and flag some harmless ones. Don't use them as the only
safeguard where a leak has legal or safety consequences.

The credential classes (`password`, `secret_key`, `jwt_token`, `private_key`, `hostname_or_endpoint`) are **experimental**
in this preview and miss many values, especially in config and log lines. Don't use the models as a secret scanner.
See the model cards for details.

## Support

Provided as is, without support or warranty. Bug reports are welcome as issues; there is no guaranteed response.

## License

Apache-2.0, for the code here and for the model weights. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
