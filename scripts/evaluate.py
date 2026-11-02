"""Study 1 / Study 4: trích biểu diễn từ mô hình mở + tính số đo (suy luận, GPU; có resume).

python scripts/evaluate.py study1_gate_g1      # cổng G1: XLM-R base, Qwen3-1.7B, OLMoE trên FLORES
python scripts/evaluate.py study1              # 15 model + random-init; FLORES và/hoặc NTREX theo DATASETS_TO_RUN
python scripts/evaluate.py study4_pilot        # Study 4: truy hồi mức từ cho XLM-R large
python scripts/evaluate.py study4 -o RUN_MODELS=[qwen3_8b]
"""

import argparse

from hourglass.cli import run
from hourglass.config import compose

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", default="study1")
    ap.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    study, params = compose(a.experiment, a.override)
    if study == "study1_open_models":
        stages = [f"run_{ds}" for ds in ("flores", "ntrex") if ds in params["DATASETS_TO_RUN"]] + ["status"]
    elif study == "study4_frequency":
        stages = ["run_models"]  # stage này tự in show_status() ở cuối
    else:
        ap.error(f"{a.experiment} thuộc {study}; dùng scripts/train.py cho Study 2a")
    run(a.experiment, stages, a.override)
