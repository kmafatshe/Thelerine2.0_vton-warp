#!/usr/bin/env python3
"""Execute the Colab notebook's code cells locally, against dummy data.

    python scripts/make_dummy_dataset.py --out /tmp/nb_data --count 10
    python scripts/test_notebook.py . /tmp/nb_data /tmp/nb_out

Cells tagged `colab-only` (Drive mount, git clone) are skipped and the training
runs are shrunk to a few steps. Everything else runs for real: the variable flow
between cells, the argument builders, and every subprocess invocation.

This exists because a notebook is code that nobody runs until it is in front of
a user. Two bugs shipped that this would have caught immediately: a cell that
used `!cmd {VAR}` — IPython leaves the literal text `{VAR}` in place when the
name is undefined, so a skipped cell surfaced as a baffling argparse error — and
a cell that appended to a global argument list, so re-running it passed the same
flag twice.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

REPO = Path(sys.argv[1])
DATA = Path(sys.argv[2])
OUT = Path(sys.argv[3])

SUBSTITUTIONS = [
    ("DATA_ROOT = Path('/content/drive/MyDrive/thelerineAI/Itekanye1.1_TripletDataset')",
     f"DATA_ROOT = Path({str(DATA)!r})"),
    ("OUTPUT_ROOT = Path('/content/drive/MyDrive/thelerine_ai_outputs/vton_warp')",
     f"OUTPUT_ROOT = Path({str(OUT)!r})"),
    ("WARP_STEPS = 12000", "WARP_STEPS = 20"),
    ("TRYON_STEPS = 15000", "TRYON_STEPS = 6"),
    ("BATCH_SIZE = 8", "BATCH_SIZE = 2"),
    ("'--limit', 5", "'--limit', 2"),
    ("TRYON_STEPS + 4000", "TRYON_STEPS + 4"),
    ("PARSE_SOURCE = 'segmentation'", "PARSE_SOURCE = 'segmentation'"),
    ("'/content/dataset_check.png'", f"'{OUT}/dataset_check.png'"),
    # Skip the VGG download in a smoke run.
    ("'train.num_workers=2',", "'train.num_workers=0', 'loss.perceptual=0.0',"),
]

notebook = json.loads((REPO / "notebooks/colab_train.ipynb").read_text())
namespace = {"__name__": "__main__"}
sys.path.insert(0, str(REPO))
import os
os.chdir(REPO)

executed = []
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    if "colab-only" in cell.get("metadata", {}).get("tags", []):
        print(f"--- cell {index}: SKIPPED (colab-only)")
        continue

    source = "".join(cell["source"])
    for old, new in SUBSTITUTIONS:
        source = source.replace(old, new)

    print(f"--- cell {index}")
    try:
        exec(compile(source, f"cell{index}", "exec"), namespace)
    except Exception as error:
        print(f"\n!!! cell {index} FAILED: {type(error).__name__}: {error}")
        print(source)
        raise SystemExit(1)
    executed.append((index, source))

    # The display helper needs IPython; stub it once it exists.
    if "def show(" in source:
        namespace["show"] = lambda p: print(f"[show] {p} exists={Path(p).exists()}")

# Idempotency: re-running the setup and scheme cells must not change the
# arguments. A previous version appended to a global list and doubled a flag.
print("\n=== idempotency check: re-running setup + scheme cells ===")
before = (namespace["data_args"](), namespace["check_args"]())
for index, source in executed:
    if "def data_args()" in source or "LABEL_SCHEME = 'cihp'" in source:
        exec(compile(source, f"cell{index}-rerun", "exec"), namespace)
after = (namespace["data_args"](), namespace["check_args"]())

if before != after:
    print(f"!!! NOT IDEMPOTENT\n  before: {before}\n  after:  {after}")
    raise SystemExit(1)
print(f"stable: {after}")
print("\n=== notebook executed cleanly ===")
