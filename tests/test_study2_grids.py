"""Study 2a: lưới chính chạy đủ mọi cấu hình, không bị cổng (G2 / pilot) chặn hay lọc."""

import pytest

pytest.importorskip("torch")

MAIN_GRIDS = {"p1p2": 108, "balanced": 24, "moe_diff": 36, "p3s2": 180, "s4": 24, "s5": 30, "heldout_p4": 48}


def test_main_grids_attempt_every_run_without_gate():
    from hourglass.study2_synthetic import load

    ws = load("study2a_grids", verbose=False)
    assert ws.RUN_GRIDS == list(MAIN_GRIDS)
    # chưa có kết quả cổng G2 / pilot nào: run_one được gọi cho MỌI cấu hình (thay bằng stub, không huấn luyện)
    seen = []
    ws.run_one = lambda cfg: seen.append(cfg["grid"]) or "ok"
    ws.run_stage("run_grids")
    assert {g: seen.count(g) for g in MAIN_GRIDS} == MAIN_GRIDS
    assert len(set(map(ws.run_id, (c for g in MAIN_GRIDS for c in ws.grid_configs(g))))) == sum(MAIN_GRIDS.values())
