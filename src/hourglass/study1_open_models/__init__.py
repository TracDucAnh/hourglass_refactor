"""Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6).

Mã giữ nguyên văn từ notebooks/archive/study-1.ipynb, tách thành các file theo chức năng và nạp vào một
namespace chung bằng ``load()`` (xem ``hourglass.workspace``).

Ví dụ::

    from hourglass.study1_open_models import load
    ws = load("study1", gpu=False)   # nạp lib
    ws.run_stage("refresh_router_js")
"""
from hourglass.config import compose
from hourglass.workspace import Step, load_workspace

NOTEBOOK = "study-1.ipynb"
DEFAULT_EXPERIMENT = "study1"
# cell code của notebook gốc không chuyển thành mã (đối chiếu trong tests/test_notebook_parity.py)
DROPPED_CELLS = {
    0: "cài package (%uv pip) + mô tả notebook -> pyproject.toml, docs/studies/study1.md",
    11: "ví dụ chạy riêng một model (đã comment) + show_status() trùng stage `status`",
    12: "%uv pip install numpy==1.26.4 -> pyproject.toml",
    22: "cell rỗng",
}

STEPS = [
    Step("config", "lib", "config.py", gpu=False,
         desc="Config & đường dẫn lưu trữ (Cell A1)", cells=(1,)),
    Step("registry", "lib", "registry.py", gpu=False,
         desc="Registry 15 model + random-init baseline (Cell A2)", cells=(2,)),
    Step("data", "lib", "data.py", gpu=False,
         desc="Dữ liệu song song FLORES-200 devtest + NTREX-128 (Cell A3)", cells=(3,)),
    Step("models", "lib", "models.py", gpu=True,
         desc="Torch, load model pretrained / random-init (Cell B1)", cells=(4,)),
    Step("extract", "lib", "extract.py", gpu=True,
         desc="Trích xuất biểu diễn theo tầng (Cell B2)", cells=(5,)),
    Step("metrics", "lib", "metrics.py", gpu=True,
         desc="Số đo M1–M4 + J_k (Cell B3)", cells=(6,)),
    Step("runner", "lib", "runner.py", gpu=False,
         desc="Lưu / đọc kết quả, resume, vòng chạy (Cell C1)", cells=(7,)),
    Step("analysis", "lib", "analysis.py", gpu=False,
         desc="Bootstrap CI, đặc trưng waist, bảng P1 / S6 (Cell D1)", cells=(13,)),
    Step("plot_style", "lib", "plotting/style.py", gpu=False,
         desc="Quy ước hình (Cell D2)", cells=(14,)),
    Step("plot_family", "lib", "plotting/family.py", gpu=False,
         desc="Hình theo nhóm kiến trúc (Cell D3)", cells=(15,)),
    Step("plot_stack", "lib", "plotting/stack.py", gpu=False,
         desc="Hình xếp chồng D/G/CKA + so sánh FLORES | NTREX (Cell D3b)", cells=(16,)),
    Step("plot_moe", "lib", "plotting/moe.py", gpu=False,
         desc="MoE: J_k routing (Cell D4)", cells=(17,)),
    Step("plot_diffusion", "lib", "plotting/diffusion.py", gpu=False,
         desc="Diffusion: mức che t (Cell D5)", cells=(18,)),
    Step("plot_summary", "lib", "plotting/summary.py", gpu=False,
         desc="Tóm tắt waist rho*, S, w (Cell D6)", cells=(19,)),
    Step("report", "lib", "report.py", gpu=False,
         desc="Tạo lại toàn bộ bảng + hình (Cell D7)", cells=(20,)),
    Step("check_data", "stage", "stages/check_data.py", gpu=False,
         desc="Tải + kiểm tra nhanh dữ liệu (cuối Cell A3)", cells=(3,)),
    Step("status", "stage", "stages/status.py", gpu=False,
         desc="Bảng trạng thái các run (Cell C2)", cells=(8,)),
    Step("run_flores", "stage", "stages/run_flores.py", gpu=True,
         desc="CHẠY FLORES-200: 15 model + random-init (Cell C3)", cells=(9,)),
    Step("run_ntrex", "stage", "stages/run_ntrex.py", gpu=True,
         desc="CHẠY NTREX-128 (Cell C4)", cells=(10,)),
    Step("figures_stack", "stage", "stages/figures_stack.py", gpu=False,
         desc="Hình xếp chồng + MoE J_k + diffusion theo dataset (cuối Cell D3b)", cells=(16,)),
    Step("analyze_flores", "stage", "stages/analyze_flores.py", gpu=False,
         desc="Bảng + hình FLORES (cuối Cell D7)", cells=(20,)),
    Step("analyze_ntrex", "stage", "stages/analyze_ntrex.py", gpu=False,
         desc="Bảng + hình NTREX (cuối Cell D7)", cells=(20,)),
    Step("refresh_router_js", "stage", "stages/refresh_router_js.py", gpu=True,
         desc="Tính lại J_k của các model MoE (cell cuối notebook)", cells=(21,)),
]


def load(experiment=DEFAULT_EXPERIMENT, overrides=(), stages=(), gpu=True, params=None, verbose=True):
    """Nạp workspace của study. ``params`` (dict) bỏ qua việc đọc YAML; ``gpu=False`` bỏ các lib cần torch."""
    if params is None:
        study, params = compose(experiment, overrides)
        if study != __name__.rsplit(".", 1)[-1]:
            raise ValueError(f"experiment {experiment} thuộc study {study}, không phải {__name__}")
    return load_workspace(__name__, STEPS, params, stages=stages, gpu=gpu, verbose=verbose)
