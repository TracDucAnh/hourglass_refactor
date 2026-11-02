"""Config YAML: mọi experiment ghép được, mọi khoá đều được study tương ứng đọc (bắt lỗi gõ sai tên)."""

import os
import re
from pathlib import Path

import pytest

from hourglass.cli import STUDIES
from hourglass.config import compose, list_experiments
from hourglass.workspace import GLOBAL_KEYS

SRC = Path(__file__).resolve().parents[1] / "src" / "hourglass"


def _param_names(study):
    pat = re.compile(r'param\("([A-Z0-9_]+)"')
    return {m for f in (SRC / study).rglob("*.py") for m in pat.findall(f.read_text(encoding="utf-8"))}


@pytest.mark.parametrize("experiment", list_experiments())
def test_experiment_composes_and_every_key_is_used(experiment):
    study, params = compose(experiment)
    assert study in STUDIES
    unknown = set(params) - _param_names(study) - GLOBAL_KEYS
    assert not unknown, f"{experiment}: khoá không study nào đọc: {sorted(unknown)}"


def test_paths_follow_hourglass_root():
    _, p = compose("study1")
    root = os.environ["HOURGLASS_ROOT"]
    assert p["ROOT"] == root
    assert p["STUDY_DIR"] == f"{root}/outputs/study1"
    assert p["ANALYSIS_DIR"] == f"{root}/results/study1"


def test_cli_overrides_are_parsed_as_yaml():
    _, p = compose("study1", ["RUN_MODELS=[xlmr_base,olmoe]", "N_SENT=200", "FORCE_RERUN=true"])
    assert p["RUN_MODELS"] == ["xlmr_base", "olmoe"]
    assert p["N_SENT"] == 200 and p["FORCE_RERUN"] is True


def test_experiment_inherits_base():
    _, base = compose("study2a")
    _, gate = compose("study2a_gate_g2")
    assert gate["RUN_GRIDS"] == ["gate_g2"]
    assert {k: v for k, v in gate.items() if k != "RUN_GRIDS"} == {k: v for k, v in base.items() if k != "RUN_GRIDS"}
