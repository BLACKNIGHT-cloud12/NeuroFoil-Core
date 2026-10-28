"""Smoke-test: execute the v2 notebook code cells sequentially against the local dataset."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

nb = json.loads((PROJECT_ROOT / "notebooks" / "neurofoil_surrogate_training_v2.ipynb").read_text(encoding="utf-8"))
ns = {"__name__": "__main__"}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    print(f"--- cell {i} ---", flush=True)
    exec(compile(src, f"<cell {i}>", "exec"), ns)
print("SMOKE TEST COMPLETE")
