"""Study 2a: lưới chính chạy đủ mọi cấu hình, không bị cổng (G2 / pilot) chặn hay lọc."""

import pytest

pytest.importorskip("torch")

MAIN_GRIDS = {"p1p2": 108, "moe_diff": 36}


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


def _run_ids(experiment, overrides=()):
    from hourglass.study2_synthetic import load

    ws = load(experiment, overrides, verbose=False)
    seen = []
    ws.run_one = lambda cfg: seen.append((cfg["family"], ws.run_id(cfg))) or "ok"
    ws.run_stage("run_grids")
    return seen, ws


def test_family_experiments_partition_main_grids():
    """4 experiment theo họ chạy đúng họ của mình và gộp lại = gate_g2 + toàn bộ lưới chính, không trùng, không sót."""
    _, ws = _run_ids("study2a_grids")
    want = {ws.run_id(c) for g in ["gate_g2", *MAIN_GRIDS] for c in ws.grid_configs(g)}
    got = []
    for fam in ("decoder", "encoder", "moe", "diffusion"):
        seen, _ = _run_ids(f"study2a_{fam}")
        assert seen and {f for f, _ in seen} == {fam}
        got += [r for _, r in seen]
    assert len(got) == len(set(got)) and set(got) == want


def test_shards_partition_a_family():
    full, _ = _run_ids("study2a_encoder")
    parts = [_run_ids("study2a_encoder", [f"SHARD={k}/3"])[0] for k in range(3)]
    ids = [r for p in parts for _, r in p]
    assert len(ids) == len(set(ids)) and sorted(ids) == sorted(r for _, r in full)
