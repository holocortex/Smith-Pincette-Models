"""A local MCP server so agents (Claude Code, Cursor, ...) can detect and mask personal data without it leaving the machine.

    pip install "smith-pincette[mcp] @ git+https://github.com/holocortex/Smith-Pincette-Models"
    python examples/06_mcp_server.py            # stdio transport

Example client config (claude_desktop_config.json / .mcp.json):
    {"mcpServers": {"smith-pincette": {"command": "python", "args": ["/path/to/examples/06_mcp_server.py"]}}}
"""

import os
from dataclasses import asdict

from mcp.server.mcpserver import MCPServer

from smith_pincette import SmithPincette

model = SmithPincette.from_pretrained(os.getenv("SMITH_PINCETTE_MODEL", "Holocortex/Smith-Pincette-Nitro"), revision="v0.5-preview")
mcp = MCPServer("smith-pincette")


@mcp.tool()
def detect_personal_data(text: str, labels: list[str] | None = None) -> list[dict]:
    """Find personal data (names, contact details, addresses, IDs, bank details, ...) in text. Returns start/end character offsets, label, score and the matched text."""
    return [asdict(s) for s in model.detect(text, labels=labels)]


@mcp.tool()
def mask_personal_data(text: str, labels: list[str] | None = None) -> str:
    """Return text with every detected piece of personal data replaced by [LABEL]."""
    return model.mask(text, labels=labels)


if __name__ == "__main__":
    mcp.run()
