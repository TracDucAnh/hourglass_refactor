"""Mã trong src/hourglass/study*/ phải tương đương (theo AST) với code cell của notebook gốc.

Cho phép đúng ba khác biệt có chủ đích:
  1. ``X = param("X", mặc_định)`` ở file config  ==  ``X = mặc_định`` trong notebook;
  2. ``from hourglass.common.m import f``          ==  ``def f`` giống hệt trong notebook;
  3. các cell trong ``DROPPED_CELLS`` (cài package, cell rỗng, ví dụ đã comment).
So sánh bằng ast.dump nên bỏ qua comment, khoảng trắng, số dòng.
"""

import ast
import importlib
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ["study1_open_models", "study2_synthetic", "study4_frequency"]


class _UnwrapParam(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "param" and len(node.args) == 2:
            return node.args[1]
        return node


def _common_defs():
    defs = {}
    for f in (ROOT / "src/hourglass/common").glob("*.py"):
        for node in ast.parse(f.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef):
                defs[(f.stem, node.name)] = ast.dump(node)
    return defs


def _notebook_cells(nb_name):
    nb = json.loads((ROOT / "notebooks/archive" / nb_name).read_text(encoding="utf-8"))
    return {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}


def _dumps(src):
    return [ast.dump(n) for n in ast.parse(src).body]


def _package_dumps(pkg_dir, step, common):
    out = []
    for node in ast.parse((pkg_dir / step.file).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hourglass.common."):
            mod = node.module.rsplit(".", 1)[-1]
            out += [common[(mod, a.name)] for a in node.names]
        else:
            out.append(ast.dump(_UnwrapParam().visit(node)))
    return out


@pytest.mark.parametrize("study", STUDIES)
def test_package_matches_notebook(study):
    mod = importlib.import_module(f"hourglass.{study}")
    pkg_dir = Path(mod.__file__).parent
    cells = _notebook_cells(mod.NOTEBOOK)
    common = _common_defs()

    # mọi code cell đều được dùng hoặc được khai báo bỏ có lý do
    used = {c for s in mod.STEPS for c in s.cells}
    assert set(cells) == used | set(mod.DROPPED_CELLS), set(cells) ^ (used | set(mod.DROPPED_CELLS))
    assert not used & set(mod.DROPPED_CELLS)

    nb = Counter(d for i, src in cells.items() if i not in mod.DROPPED_CELLS for d in _dumps(src))
    pkg = Counter()
    for step in mod.STEPS:
        dumps = Counter(_package_dumps(pkg_dir, step, common))
        own = Counter(d for i in step.cells for d in _dumps(cells[i]))
        assert not dumps - own, f"{step.file}: có câu lệnh không đến từ cell {step.cells}"
        pkg += dumps
    assert pkg == nb, f"lệch: thiếu {len(nb - pkg)}, thừa {len(pkg - nb)} câu lệnh cấp module"
