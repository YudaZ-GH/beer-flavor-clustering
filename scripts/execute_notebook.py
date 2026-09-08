"""Execute the project notebook from a clean kernel and save verified outputs."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat
from nbclient import NotebookClient


NOTEBOOK_PATH = Path("Stat 437 final project.ipynb").resolve()

with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
    notebook = nbformat.read(handle, as_version=4)

for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        ast.parse(cell.source, filename=f"notebook-cell-{index}")

client = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    allow_errors=False,
)
client.execute()

with NOTEBOOK_PATH.open("w", encoding="utf-8", newline="\n") as handle:
    nbformat.write(notebook, handle)

print(f"Executed and saved {NOTEBOOK_PATH.name} from a clean kernel.")
