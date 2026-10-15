"""Inline results/explorer_data.json into docs/index.html (static page, no server, no network)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARK = "/*__DATA__*/null"


def main() -> None:
    template = (ROOT / "scripts" / "explorer_template.html").read_text(encoding="utf-8")
    data = json.loads((ROOT / "results" / "explorer_data.json").read_text(encoding="utf-8"))
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    assert MARK in template
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(template.replace(MARK, payload), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
