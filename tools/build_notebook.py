"""Convert a percent-format Python script into a Jupyter notebook.

The source lives as a plain `.py` in the "percent" format that jupytext, VS Code
and PyCharm all understand:

    # %% [markdown]
    # Some prose, one `#` per line.

    # %%
    code_goes_here()

Keeping the source runnable as a script is the point: the notebook can be
smoke-tested with `python tools/frozen_head_source.py` before it is ever
converted, which catches syntax and import errors that would otherwise only
surface on Kaggle after a ten-minute model download.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_MARKER = "# %%"
MARKDOWN_MARKER = "# %% [markdown]"


def _strip_comment(line: str) -> str:
    """Turn one `# text` source line back into its markdown text."""
    if line.startswith("# "):
        return line[2:]
    if line == "#":
        return ""
    return line


def split_cells(text: str) -> list[tuple[str, str]]:
    """Split percent-format source into (cell_type, source) pairs."""
    cells: list[tuple[str, str]] = []
    kind = "code"
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip("\n")
        if body:
            cells.append((kind, body))

    for line in text.splitlines():
        if line.startswith(CODE_MARKER):
            flush()
            kind = "markdown" if line.startswith(MARKDOWN_MARKER) else "code"
            buffer = []
            continue
        buffer.append(line)

    flush()
    return cells


def build(cells: list[tuple[str, str]]) -> dict:
    """Wrap cells in the minimal nbformat 4 envelope Kaggle accepts."""
    out = []
    for kind, body in cells:
        if kind == "markdown":
            body = "\n".join(_strip_comment(l) for l in body.splitlines()).strip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": body.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)

    return {
        "cells": out,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    cells = split_cells(source.read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build(cells), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    n_md = sum(1 for k, _ in cells if k == "markdown")
    print(f"{target}: {len(cells)} cells ({n_md} markdown, {len(cells) - n_md} code)")


if __name__ == "__main__":
    main()
