"""Ghép config YAML trong ``configs/`` thành một dict tham số phẳng cho một thí nghiệm.

Thứ tự ghép (sau đè trước):
  1. ``configs/config.yaml``              — đường dẫn gốc, SEED
  2. các file trong ``defaults:`` của experiment (đệ quy), vd. ``data/flores_ntrex12``, ``model/tiny_lm``
  3. ``configs/experiment/<tên>.yaml``    — mỗi file = một thí nghiệm trong paper
  4. override dòng lệnh ``KEY=VALUE``      — giá trị đọc như YAML, vd. ``RUN_MODELS=[xlmr_base,olmoe]``

Khoá viết HOA = tên hằng trong code của study (đọc bằng ``param("TÊN", mặc_định)``); khoá ``study`` chọn package.
"""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("HOURGLASS_CONFIG_DIR", REPO_ROOT / "configs"))


def _load(rel: str):
    path = CONFIG_DIR / f"{rel}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"không thấy config {path}")
    node = OmegaConf.load(path)
    defaults = list(node.pop("defaults", []) or [])
    return OmegaConf.merge(*[_load(d) for d in defaults], node) if defaults else node


def list_experiments() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml"))


def compose(experiment: str, overrides=()) -> tuple[str, dict]:
    """Trả về (tên package study, dict tham số đã nội suy)."""
    cfg = OmegaConf.merge(
        {"REPO_ROOT": str(REPO_ROOT)},
        _load("config"),
        _load(f"experiment/{experiment}"),
        OmegaConf.from_dotlist(list(overrides)),
    )
    params = OmegaConf.to_container(cfg, resolve=True)
    study = params.pop("study", None)
    if not study:
        raise ValueError(f"experiment {experiment} thiếu khoá `study`")
    return study, params
