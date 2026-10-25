import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isl.translate import translate  # noqa: E402

CASES = json.loads((ROOT / "testcases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", [c for c in CASES if not c.get("known_limitation")],
                         ids=lambda c: f"{c['id']}-{c['sentence']}")
def test_gloss(case):
    """Paper rules only (no sign lookup)."""
    assert " ".join(translate(case["sentence"])) == case["expected"]
