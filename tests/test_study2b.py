"""Study 2b: lưới (run_id, twin dùng chung), số đo ở trường hợp biên, và cả đường ống trên dữ liệu toy (CPU)."""

import os

import numpy as np
import pytest

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    from hourglass.study3_natural import load

    os.environ["HOURGLASS_ROOT"] = str(tmp_path_factory.mktemp("root2b"))
    return load("study2b_smoke", verbose=False)


def test_grids_share_twins_and_runs(ws):
    from hourglass.study3_natural import load

    real = load("study2b", gpu=False, verbose=False)
    p3 = real.grid_configs("p3")
    ids = [real.run_id(c) for c in p3]
    assert len(ids) == len(set(ids))
    twins = [c for c in p3 if c["kind"] == "twin"]
    joints = [c for c in p3 if c["kind"] == "joint"]
    # twin dùng chung cho mọi |L| và mọi thứ tự lồng: 8 ngôn ngữ x 3 d x 3 seed
    assert len(twins) == 8 * len(real.D_P3) * real.N_SEEDS
    assert len(joints) == len(real.D_P3) * len(real.L_LIST) * len(real.NESTINGS) * real.N_SEEDS
    # mỗi run joint có đủ twin cùng d, frac, tokenizer, seed
    tw = set(ids)
    for c in joints:
        for lang in real.langs_of(c):
            t = real.make_cfg("p3", c["family"], "twin", c["d"], lang=lang, frac=c["frac"], tok=c["tok"],
                              seed=c["seed"])
            assert real.run_id(t) in tw
    # pilot và p3 trùng cấu hình -> cùng run_id (không huấn luyện lại)
    assert {real.run_id(c) for c in real.grid_configs("pilot_data")} & set(ids)
    # decoder lưu vào thư mục Study 4 đọc; twin bắt đầu bằng "twin" (Study 4 bỏ qua)
    assert all(os.path.basename(real.ckpt_path(c)).startswith("twin") for c in twins)
    assert all(real.ckpt_path(c).startswith(real.CKPT_DIR) for c in joints)


def test_measures_on_edge_cases(ws):
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(1, 1, 200, 32)).astype(np.float32)
    same, _ = ws.pair_metrics(np.concatenate([Z, Z], 1))
    indep, _ = ws.pair_metrics(np.concatenate([Z, rng.normal(size=Z.shape).astype(np.float32)], 1))
    assert abs(same["D"][0, 0]) < 1e-6 and abs(same["G"][0, 0] - 1) < 1e-6 and abs(same["A"][0, 0] - 1) < 1e-6
    assert abs(indep["D"][0, 0] - 2) < 0.15 and abs(indep["G"][0, 0]) < 0.1
    # A của mô hình chung và A_indep giữa hai twin dùng cùng thủ tục
    assert np.isclose(ws.twin_A(Z[:, 0], Z[:, 0])[0], same["A"][0, 0])


@pytest.mark.slow
def test_smoke_pipeline_end_to_end(ws):
    for stage in ("prepare_data", "build_tokens", "run_grids", "analyze"):
        ws.run_stage(stage)
    import pandas as pd

    summary = pd.read_csv(f"{ws.ANALYSIS_DIR}/hypotheses_summary.csv")
    assert set(summary.hypothesis) == {"P3", "S2", "S4", "S5", "P4"}
    feats = pd.read_csv(f"{ws.ANALYSIS_DIR}/features_joint_runs.csv")
    assert feats.dA_waist.notna().any()  # ΔA tính được từ twin
    assert {"decoder", "moe", "diffusion"} <= set(feats.family)
    # resume: chạy lại không huấn luyện gì
    assert set(ws.run_grid("smoke").values()) == {"skipped"}
