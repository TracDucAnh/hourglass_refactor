"""Study 2a: huấn luyện mô hình tí hon trên các lưới ngôn ngữ tổng hợp (GPU).

    python scripts/train.py study2a_pilot_conv pilot_report    # lưới pilot hội tụ + báo cáo chọn BATCH/SEQ_MULT
    python scripts/train.py study2a_gate_g2 gate_report        # cổng G2
    python scripts/train.py study2a_grids                      # toàn lưới chính
    python scripts/train.py study2a_smoke gate_report -o FORCE_RERUN=true

Chạy stage `run_grids` (mọi lưới trong RUN_GRIDS của experiment, có resume) rồi các stage thêm (nếu có).
"""

import argparse

from hourglass.cli import run
from hourglass.config import compose

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", default="study2a")
    ap.add_argument("extra_stages", nargs="*", help="vd. gate_report, pilot_report, status")
    ap.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    study, _ = compose(a.experiment, a.override)
    if study != "study2_synthetic":
        ap.error(f"{a.experiment} thuộc {study}; train.py chỉ dành cho Study 2a (dùng evaluate.py cho Study 1 / 4)")
    run(a.experiment, ["run_grids", *a.extra_stages], a.override)
