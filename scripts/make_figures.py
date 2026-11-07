"""Sinh lại bảng (results/) và hình (paper/figures/) từ kết quả đã chạy — không chạy lại mô hình.

    python scripts/make_figures.py                 # cả ba study
    python scripts/make_figures.py study1 study2a
    python scripts/make_figures.py study2a_smoke   # bất kỳ experiment nào

Study 1 và 2a chỉ cần CPU. Study 4 nạp lại state dữ liệu từ cache (tách từ, tần suất, ITEMS) và cần torch
(bootstrap GAM chạy trên DEVICE, CPU cũng được nhưng chậm).
"""

import argparse

from hourglass.cli import run
from hourglass.config import compose

STAGES = {
    "study1_open_models": (["analyze_flores", "analyze_ntrex", "figures_stack"], False),
    "study2_synthetic": (["analyze"], False),
    "study3_natural": (["analyze"], False),
    "study4_frequency": (
        ["s3", "robustness", "mantel", "capacity", "evidence", "shape_v6", "figures", "figures_v6"],
        True,
    ),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*", default=["study1", "study2a", "study4"])
    ap.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    for exp in a.experiments:
        study, _ = compose(exp, a.override)
        stages, gpu = STAGES[study]
        print(f"===== {exp}: {stages}")
        run(exp, stages, a.override, gpu=gpu)
