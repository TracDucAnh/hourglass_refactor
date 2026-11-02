"""Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6).

Mã giữ nguyên văn từ notebooks/archive/study-2a.ipynb, tách thành các file theo chức năng và nạp vào một
namespace chung bằng ``load()`` (xem ``hourglass.workspace``).

Ví dụ::

    from hourglass.study2_synthetic import load
    ws = load("study2a", gpu=False)   # nạp lib
    ws.run_stage("analyze")
"""
from hourglass.config import compose
from hourglass.workspace import Step, load_workspace

NOTEBOOK = "study-2a.ipynb"
DEFAULT_EXPERIMENT = "study2a"
# cell code của notebook gốc không chuyển thành mã (đối chiếu trong tests/test_notebook_parity.py)
DROPPED_CELLS = {
    0: "cài package (%uv pip) -> pyproject.toml",
    42: "cell rỗng",
}

STEPS = [
    Step("config", "lib", "config.py", gpu=False,
         desc="Config, đường dẫn, kích thước lưới, giao thức huấn luyện v2 (Cell A1)", cells=(3,)),
    Step("semantics", "lib", "synth/semantics.py", gpu=False,
         desc="Ngữ nghĩa C: kho khái niệm + PCFG R00–R28 (Cell S1)", cells=(6,)),
    Step("languages", "lib", "synth/languages.py", gpu=False,
         desc="Ngôn ngữ + renderer g_l, luật T01–T10 (Cell S2)", cells=(7,)),
    Step("parser", "lib", "synth/parser.py", gpu=False,
         desc="Parser nghịch kiểm giả định (G) (Cell S3)", cells=(8,)),
    Step("bpe", "lib", "synth/bpe.py", gpu=False,
         desc="Tokenizer BPE chung (Cell S4)", cells=(9,)),
    Step("world", "lib", "synth/world.py", gpu=False,
         desc="Dựng 'thế giới': corpus + tập đo, có cache (Cell S5)", cells=(10,)),
    Step("checks", "lib", "synth/checks.py", gpu=False,
         desc="Tiện ích kiểm thử bộ sinh (Cell T0)", cells=(12,)),
    Step("models", "lib", "models.py", gpu=True,
         desc="Torch + mô hình 4 họ TinyLM (Cell B1)", cells=(20,)),
    Step("training", "lib", "training.py", gpu=True,
         desc="Tác vụ LM / SEM / SURF + huấn luyện + giám sát hội tụ (Cell B2)", cells=(21,)),
    Step("extract", "lib", "extract.py", gpu=True,
         desc="Trích biểu diễn gộp theo từ (Cell B3)", cells=(22,)),
    Step("metrics", "lib", "metrics.py", gpu=True,
         desc="Số đo M1–M4 (Cell B4)", cells=(23,)),
    Step("logit_lens", "lib", "logit_lens.py", gpu=True,
         desc="Logit-lens: ngôn ngữ trục (S4) (Cell B5)", cells=(24,)),
    Step("grids", "lib", "grids.py", gpu=False,
         desc="Định nghĩa lưới + run_id (Cell C1)", cells=(26,)),
    Step("runner", "lib", "runner.py", gpu=False,
         desc="Lưu / đọc, đo một mô hình, sinh đôi, vòng chạy (Cell C2)", cells=(27,)),
    Step("gate", "lib", "gate.py", gpu=False,
         desc="Báo cáo cổng G2 + dự toán thời gian (Cell C5)", cells=(31,)),
    Step("pilot", "lib", "pilot.py", gpu=False,
         desc="Báo cáo pilot hội tụ (Cell C5b)", cells=(32,)),
    Step("features", "lib", "analysis/features.py", gpu=False,
         desc="Tải kết quả + đặc trưng waist (Cell D1)", cells=(35,)),
    Step("stats", "lib", "analysis/stats.py", gpu=False,
         desc="Tiện ích thống kê (Cell D2)", cells=(36,)),
    Step("hypotheses", "lib", "analysis/hypotheses.py", gpu=False,
         desc="Kiểm định P1, P2, P3, S2–S6 (Cell D3)", cells=(37,)),
    Step("plot_style", "lib", "plotting/style.py", gpu=False,
         desc="Style hình (Cell D4)", cells=(38,)),
    Step("plot_figures", "lib", "plotting/figures.py", gpu=False,
         desc="Hình theo giả thuyết (Cell D5)", cells=(39,)),
    Step("report", "lib", "report.py", gpu=False,
         desc="Báo cáo kết luận (Cell D6)", cells=(40,)),
    Step("g0_generator_tests", "stage", "stages/g0_generator_tests.py", gpu=False,
         desc="Cổng G0: testcase bộ sinh + số đo (Cell T1–T6)", cells=(13, 14, 15, 16, 17, 18)),
    Step("status", "stage", "stages/status.py", gpu=False,
         desc="Trạng thái hiện tại (Cell C3)", cells=(28,)),
    Step("diag_embedding", "stage", "stages/diag_embedding.py", gpu=True,
         desc="Chẩn đoán U_0/log L ở embedding (Cell C3b)", cells=(29,)),
    Step("run_grids", "stage", "stages/run_grids.py", gpu=True,
         desc="Chạy các lưới trong RUN_GRIDS (Cell C4)", cells=(30,)),
    Step("gate_report", "stage", "stages/gate_report.py", gpu=False,
         desc="In báo cáo cổng G2 (cuối Cell C5)", cells=(31,)),
    Step("pilot_report", "stage", "stages/pilot_report.py", gpu=False,
         desc="In báo cáo pilot hội tụ (cuối Cell C5b)", cells=(32,)),
    Step("run_remaining_grids", "stage", "stages/run_remaining_grids.py", gpu=True,
         desc="Chạy các lưới còn lại theo thứ tự gợi ý (Cell C6)", cells=(33,)),
    Step("analyze", "stage", "stages/analyze.py", gpu=False,
         desc="Đặc trưng -> kiểm định -> hình -> báo cáo (Cell D7)", cells=(41,)),
]


def load(experiment=DEFAULT_EXPERIMENT, overrides=(), stages=(), gpu=True, params=None, verbose=True):
    """Nạp workspace của study. ``params`` (dict) bỏ qua việc đọc YAML; ``gpu=False`` bỏ các lib cần torch."""
    if params is None:
        study, params = compose(experiment, overrides)
        if study != __name__.rsplit(".", 1)[-1]:
            raise ValueError(f"experiment {experiment} thuộc study {study}, không phải {__name__}")
    return load_workspace(__name__, STEPS, params, stages=stages, gpu=gpu, verbose=verbose)
