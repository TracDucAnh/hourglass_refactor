"""Nạp workspace (namespace chung) và gọi vài hàm phân tích không cần GPU."""

import numpy as np
import pytest

from hourglass.config import compose
from hourglass.workspace import load_workspace


@pytest.fixture(scope="module")
def ws1():
    from hourglass.study1_open_models import load

    return load("study1", gpu=False, verbose=False)


def test_study1_gpu_libs_are_skipped(ws1):
    assert "compute_metrics" not in vars(ws1)  # lib metrics cần torch
    assert "waist_from_curves" in vars(ws1) and len(ws1.MODELS) == 15


def test_study1_waist_features_on_inverted_u(ws1):
    k = np.arange(13)
    G = np.exp(-(((k - 6) / 3.0) ** 2))
    ks, rho, S, w = (x[0] for x in ws1.curve_features(G[None]))
    assert ks == 6 and np.isclose(rho, 0.5) and S > 0 and 0 < w < 1
    assert ws1.edge_depth(G) > 0.5
    assert ws1.edge_depth(np.linspace(0, 1, 13)) == 0  # đường đơn điệu: không có eo


def test_stage_validation():
    from hourglass.study1_open_models import load

    with pytest.raises(ValueError, match="không tồn tại"):
        load("study1", stages=["khong_co"], gpu=False, verbose=False)
    with pytest.raises(ValueError, match="cần GPU"):
        load("study1", stages=["run_flores"], gpu=False, verbose=False)


def test_experiment_must_match_study():
    from hourglass.study2_synthetic import load

    with pytest.raises(ValueError, match="thuộc study"):
        load("study1", gpu=False, verbose=False)


def test_study4_config_and_registry():
    from hourglass.study4_frequency import STEPS

    _, params = compose("study4")
    ws = load_workspace("hourglass.study4_frequency", STEPS[:2], params, verbose=False)
    assert len(ws.PAIRS) == 66 and ws.PAIR_NAME[0] == "eng-fra"
    assert ws.WORK_DIR == f"{params['STUDY_DIR']}/work_full"
    assert isinstance(ws.EDGE_Q, tuple) and isinstance(ws.LEMMA_LANGS, set)  # giữ kiểu như notebook
    assert {"xlmr_large", "qwen3_8b"} <= set(ws.MODELS)
