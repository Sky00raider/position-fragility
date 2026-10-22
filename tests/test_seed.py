import subprocess
import sys
from pathlib import Path

from src.store import cell_seed

ROOT = Path(__file__).resolve().parent.parent


def test_seed_stable_across_processes():
    code = "from src.store import cell_seed; print(cell_seed('run', '3hop1__1', 'inject', 2, 'in_context'))"
    outs = {subprocess.check_output([sys.executable, "-c", code], cwd=ROOT).strip() for _ in range(2)}
    assert len(outs) == 1
    assert int(outs.pop()) == cell_seed("run", "3hop1__1", "inject", 2, "in_context")


def test_seed_differs_by_cell():
    assert cell_seed("r", "i", "inject", 1, "in_context") != cell_seed("r", "i", "inject", 2, "in_context")
