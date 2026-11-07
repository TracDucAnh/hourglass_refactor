"""Study 2b — Ngôn ngữ tự nhiên: eo và căn chỉnh thay đổi thế nào theo capacity và số ngôn ngữ (P3, S2, S4; P4, S5 phụ).

Khác các study khác, study này không có notebook gốc: mã viết trực tiếp ở đây, nhưng vẫn nạp vào một namespace
chung bằng ``load()`` (xem ``hourglass.workspace``) để dùng chung CLI, config và ``scripts/train.py``.

Checkpoint decoder được lưu ở ``{ROOT}/study3_natural/checkpoints/<tên>/`` (định dạng HF Llama) để Study 4 tự
đăng ký và đo S3 trên chúng; thư mục mô hình sinh đôi bắt đầu bằng ``twin`` (Study 4 bỏ qua).

Ví dụ::

    from hourglass.study3_natural import load
    ws = load("study2b_smoke", stages=["prepare_data", "build_tokens", "run_grids", "analyze"])
"""
from hourglass.config import compose
from hourglass.workspace import Step, load_workspace

DEFAULT_EXPERIMENT = "study2b"

STEPS = [
    Step("config", "lib", "config.py", desc="Config, đường dẫn, ngôn ngữ, giao thức huấn luyện"),
    Step("data", "lib", "data.py", desc="FLORES (đo), byte premium, corpus FineWeb-2 hoặc toy, lọc trùng 13-gram"),
    Step("tokenization", "lib", "tokenization.py", desc="BPE chung (và biến thể cho S4), mã hoá corpus, token -> ngôn ngữ"),
    Step("grids", "lib", "grids.py", desc="Lưới thí nghiệm + run_id (joint / twin)"),
    Step("metrics", "lib", "metrics.py", desc="M1-M2 bằng numpy: D, G, CKA, A (Procrustes-CV), đặc trưng eo"),
    Step("models", "lib", "models.py", gpu=True, desc="Llama (decoder, diffusion hai chiều) / Mixtral (MoE) theo HF"),
    Step("training", "lib", "training.py", gpu=True, desc="Dòng token, loss theo họ, vòng huấn luyện, giám sát hội tụ"),
    Step("measure", "lib", "measure.py", gpu=True, desc="Trích biểu diễn FLORES (token cuối / trung bình), M1-M2, logit-lens"),
    Step("runner", "lib", "runner.py", gpu=True, desc="Một run: huấn luyện, lưu checkpoint, đo; vòng lưới có resume"),
    Step("analysis", "lib", "analysis.py", desc="Đặc trưng eo, ΔA, kiểm định P3 / S2 / S4, P4 / S5, báo cáo, hình"),
    Step("prepare_data", "stage", "stages/prepare_data.py", desc="Tải / sinh dữ liệu: FLORES, byte premium, corpus"),
    Step("build_tokens", "stage", "stages/build_tokens.py", desc="Train tokenizer cần cho RUN_GRIDS + mã hoá corpus"),
    Step("run_grids", "stage", "stages/run_grids.py", gpu=True, desc="Chạy các lưới trong RUN_GRIDS"),
    Step("status", "stage", "stages/status.py", desc="Trạng thái các lưới"),
    Step("analyze", "stage", "stages/analyze.py", desc="Đặc trưng -> kiểm định -> hình -> báo cáo"),
]


def load(experiment=DEFAULT_EXPERIMENT, overrides=(), stages=(), gpu=True, params=None, verbose=True):
    """Nạp workspace của study. ``params`` (dict) bỏ qua việc đọc YAML; ``gpu=False`` bỏ các lib cần torch."""
    if params is None:
        study, params = compose(experiment, overrides)
        if study != __name__.rsplit(".", 1)[-1]:
            raise ValueError(f"experiment {experiment} thuộc study {study}, không phải {__name__}")
    return load_workspace(__name__, STEPS, params, stages=stages, gpu=gpu, verbose=verbose)
