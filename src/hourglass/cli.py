"""Dòng lệnh: ``python -m hourglass <lệnh>``.

list                                   liệt kê experiment trong configs/experiment/
stages  <experiment>                   liệt kê lib + stage của study tương ứng
config  <experiment> [-o K=V ...]      in tham số đã ghép
run     <experiment> [stage ...] [-o K=V ...] [--no-gpu]
                                       nạp workspace rồi chạy các stage theo thứ tự
"""

from __future__ import annotations

import argparse
import importlib
import json
import os

from hourglass.config import compose, list_experiments

STUDIES = ("study1_open_models", "study2_synthetic", "study4_frequency")


def study_module(name: str):
    if name not in STUDIES:
        raise ValueError(f"study không hợp lệ: {name}. Có: {STUDIES}")
    return importlib.import_module(f"hourglass.{name}")


def run(experiment: str, stages=(), overrides=(), gpu: bool = True):
    os.environ.setdefault("MPLBACKEND", "Agg")  # chạy không màn hình: plt.show() không mở cửa sổ
    study, params = compose(experiment, overrides)
    return study_module(study).load(params=params, stages=stages, gpu=gpu)


def main(argv=None, default_stages=()):
    ap = argparse.ArgumentParser(
        prog="python -m hourglass", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("stages", "config", "run"):
        p = sub.add_parser(name)
        p.add_argument("experiment")
        p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
        if name == "run":
            p.add_argument("stage", nargs="*", default=list(default_stages))
            p.add_argument("--no-gpu", action="store_true", help="bỏ qua lib cần torch (chỉ phân tích / vẽ)")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for e in list_experiments():
            print(f"{e:28s} -> {compose(e)[0]}")
        return
    study, params = compose(a.experiment, a.override)
    if a.cmd == "config":
        print(json.dumps(params, indent=2, ensure_ascii=False))
    elif a.cmd == "stages":
        for s in study_module(study).STEPS:
            print(f"{s.kind:5s} {s.name:22s} {'torch ' if s.gpu else '      '}{s.desc}")
    else:
        run(a.experiment, a.stage, a.override, gpu=not a.no_gpu)


if __name__ == "__main__":
    main()
