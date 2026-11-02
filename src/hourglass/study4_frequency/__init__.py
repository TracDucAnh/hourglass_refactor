"""Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên).

Mã giữ nguyên văn từ notebooks/archive/study-4.ipynb, tách thành các file theo chức năng và nạp vào một
namespace chung bằng ``load()`` (xem ``hourglass.workspace``).

Ví dụ::

    from hourglass.study4_frequency import load
    ws = load("study4")   # nạp lib
    ws.run_stage("diag_extra")
"""
from hourglass.config import compose
from hourglass.workspace import Step, load_workspace

NOTEBOOK = "study-4.ipynb"
DEFAULT_EXPERIMENT = "study4"
# cell code của notebook gốc không chuyển thành mã (đối chiếu trong tests/test_notebook_parity.py)
DROPPED_CELLS = {
    1: "cài package (%uv pip) -> pyproject.toml (extra `study4`)",
    46: "cell rỗng",
}

STEPS = [
    Step("config", "lib", "config.py", gpu=False,
         desc="Config & đường dẫn lưu trữ (Cell A1)", cells=(2,)),
    Step("registry", "lib", "registry.py", gpu=False,
         desc="Registry model core / sweep / 2b local (Cell A2)", cells=(3,)),
    Step("data", "lib", "data.py", gpu=False,
         desc="FLORES devtest (đo) + FLORES dev / NTREX (gióng hàng) (Cell A3)", cells=(4,)),
    Step("words", "lib", "words.py", gpu=False,
         desc="Tách từ + UPOS (Cell A4)", cells=(6,)),
    Step("corpus", "lib", "corpus.py", gpu=False,
         desc="STATE: gộp các nguồn đo thành một corpus song song -> TEXTS, WT", cells=(7,)),
    Step("frequency", "lib", "frequency.py", gpu=False,
         desc="Tần suất từ FineWeb-2 -> hạng (Cell A5)", cells=(8,)),
    Step("stanza_models", "lib", "stanza_models.py", gpu=False,
         desc="Tải model Stanza (tokenize, pos)", cells=(9,)),
    Step("frequency_counts", "lib", "frequency_counts.py", gpu=False,
         desc="STATE: đếm tần suất + nạp RANKS", cells=(10,)),
    Step("alignment", "lib", "alignment.py", gpu=False,
         desc="Gióng hàng Eflomal + xác nhận MUSE / Wiktionary (Cell A6)", cells=(11,)),
    Step("lemmas", "lib", "lemmas.py", gpu=False,
         desc="Lemma Stanza làm khoá tra từ điển (Cell A6a)", cells=(12,)),
    Step("items", "lib", "items.py", gpu=False,
         desc="STATE: ITEMS, PAIR_STATS, NEED, PAIR_MAIN, ITEMS_SIG", cells=(13,)),
    Step("upos_projection", "lib", "upos_projection.py", gpu=False,
         desc="Chiếu UPOS cho ngôn ngữ không có tagger (Cell A6b)", cells=(14,)),
    Step("extract", "lib", "extract.py", gpu=True,
         desc="Load model + trích biểu diễn mức từ (Cell B1)", cells=(19,)),
    Step("retrieval", "lib", "retrieval.py", gpu=True,
         desc="Truy hồi bản dịch M3 theo tầng (Cell B2)", cells=(20,)),
    Step("runner", "lib", "runner.py", gpu=True,
         desc="Lưu / đọc kết quả, resume, vòng chạy (Cell C1)", cells=(21,)),
    Step("tables", "lib", "tables.py", gpu=True,
         desc="Bảng cấp từ và cấp cặp (Cell C3)", cells=(25,)),
    Step("stats", "lib", "analysis/stats.py", gpu=True,
         desc="GAM P-spline, bootstrap trên GPU, đơn điệu / unimodal, Mantel (Cell C4)", cells=(27,)),
    Step("shape", "lib", "analysis/shape.py", gpu=True,
         desc="Kiểm định hình dạng v6 (Cell C4b)", cells=(29,)),
    Step("s3", "lib", "analysis/s3.py", gpu=True,
         desc="S3: R_k theo log-hạng (Cell C5)", cells=(30,)),
    Step("mantel", "lib", "analysis/mantel.py", gpu=True,
         desc="Mantel: căn chỉnh vs khoảng cách loại hình (Cell C6)", cells=(33,)),
    Step("capacity", "lib", "analysis/capacity.py", gpu=True,
         desc="Capacity: đỉnh r* theo cỡ model (Cell C7)", cells=(35,)),
    Step("plot_style", "lib", "plotting/style.py", gpu=True,
         desc="Quy ước hình (Cell D1)", cells=(39,)),
    Step("plot_figures", "lib", "plotting/figures.py", gpu=True,
         desc="Hình Study 4 (Cell D2)", cells=(40,)),
    Step("diag_hub", "lib", "diagnostics/hub_modes.py", gpu=True,
         desc="Chẩn đoán hubness CSLS (Cell E1)", cells=(44,)),
    Step("check_data", "stage", "stages/check_data.py", gpu=False,
         desc="Kiểm tra nhanh dữ liệu", cells=(5,)),
    Step("diag_alignment", "stage", "stages/diag_alignment.py", gpu=False,
         desc="Chẩn đoán gióng hàng / từ điển (Cell A7, A7b, A7c, A7d + cell chẩn đoán độ phủ)", cells=(15, 16, 17, 18, 23)),
    Step("status", "stage", "stages/status.py", gpu=True,
         desc="Trạng thái các model", cells=(22,)),
    Step("run_models", "stage", "stages/run_models.py", gpu=True,
         desc="CHẠY trích biểu diễn + truy hồi (Cell C2)", cells=(24,)),
    Step("waist_check", "stage", "stages/waist_check.py", gpu=True,
         desc="Kiểm tra nhanh tầng waist theo truy hồi", cells=(26,)),
    Step("s3", "stage", "stages/s3.py", gpu=True,
         desc="Chạy S3 cho mọi model đã có kết quả -> S3_ALL", cells=(31,)),
    Step("robustness", "stage", "stages/robustness.py", gpu=True,
         desc="Độ bền theo lựa chọn ngôn ngữ (Cell C5b)", cells=(32,)),
    Step("mantel", "stage", "stages/mantel.py", gpu=True,
         desc="Chạy Mantel", cells=(34,)),
    Step("capacity", "stage", "stages/capacity.py", gpu=True,
         desc="Bảng capacity + kết luận S3", cells=(36,)),
    Step("evidence", "stage", "stages/evidence.py", gpu=True,
         desc="Bảng mức độ bằng chứng (Cell C7b)", cells=(37,)),
    Step("shape_v6", "stage", "stages/shape_v6.py", gpu=True,
         desc="Hình dạng + capacity + độ bền v6 (Cell C7c)", cells=(38,)),
    Step("figures", "stage", "stages/figures.py", gpu=True,
         desc="Vẽ toàn bộ hình D2", cells=(41,)),
    Step("figures_v6", "stage", "stages/figures_v6.py", gpu=True,
         desc="Hình S3 v6 (Cell D3)", cells=(42,)),
    Step("diag_extra", "stage", "stages/diag_extra.py", gpu=True,
         desc="Chẩn đoán bổ sung v5 (Cell E2)", cells=(45,)),
]


def load(experiment=DEFAULT_EXPERIMENT, overrides=(), stages=(), gpu=True, params=None, verbose=True):
    """Nạp workspace của study. ``params`` (dict) bỏ qua việc đọc YAML; ``gpu=False`` bỏ các lib cần torch."""
    if params is None:
        study, params = compose(experiment, overrides)
        if study != __name__.rsplit(".", 1)[-1]:
            raise ValueError(f"experiment {experiment} thuộc study {study}, không phải {__name__}")
    return load_workspace(__name__, STEPS, params, stages=stages, gpu=gpu, verbose=verbose)
