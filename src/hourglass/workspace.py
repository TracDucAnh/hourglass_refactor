"""Workspace: nạp các file của một study vào MỘT namespace chung, theo thứ tự cell của notebook gốc.

Vì sao không dùng import module thông thường: ba notebook gốc chạy trong một kernel, nên hàm ở cell sau đọc
biến toàn cục do cell trước tạo ra (vd. ``WT``, ``ITEMS``, ``PAIR_STATS`` ở Study 4; ``MODELS``, ``DEVICE`` ở mọi
study). Nạp các file vào một namespace chung giữ nguyên ngữ nghĩa đó, nên mã trong ``src/hourglass/study*/`` được
giữ NGUYÊN VĂN từ notebook (kiểm bằng ``tests/test_notebook_parity.py``).

Mỗi study khai báo một danh sách ``Step``:
  - ``lib``  : định nghĩa (hàm, hằng, state dựng từ cache) — luôn được nạp, theo thứ tự.
  - ``stage``: hành động (chạy lưới, vẽ hình, chẩn đoán) — chỉ chạy khi được yêu cầu, sau khi đã nạp mọi lib.

Tham số: file config của study gọi ``param("TÊN", mặc_định)``; giá trị lấy từ YAML trong ``configs/``
(xem ``hourglass.config``), không có thì dùng mặc định của notebook.
"""

from __future__ import annotations

import builtins
import sys
import time
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# Khoá chung của configs/config.yaml: không phải study nào cũng đọc (vd. Study 2a không dùng DATA_DIR)
# -> không cảnh báo là thừa.
GLOBAL_KEYS = {"REPO_ROOT", "ROOT", "DATA_DIR", "OUTPUT_DIR", "RESULTS_DIR", "FIGURE_DIR", "HF_CACHE_DIR", "SEED"}


@dataclass(frozen=True)
class Step:
    name: str
    kind: str  # "lib" | "stage"
    file: str
    gpu: bool = False
    desc: str = ""
    cells: tuple[int, ...] = field(default_factory=tuple)  # chỉ số cell trong notebook gốc (để đối chiếu)


class Params:
    """``param(name, default)`` trong namespace: trả giá trị từ config, giữ kiểu tuple/set của mặc định."""

    def __init__(self, values: dict):
        self.values = dict(values)
        self.used: set[str] = set()

    def __call__(self, name: str, default):
        self.used.add(name)
        if name not in self.values:
            return default
        v = self.values[name]
        if isinstance(default, tuple) and isinstance(v, list):
            return tuple(v)
        if isinstance(default, (set, frozenset)) and isinstance(v, list):
            return type(default)(v)
        return v


def load_workspace(
    package: str, steps: list[Step], params: dict, stages=(), gpu: bool = True, verbose: bool = True
) -> types.ModuleType:
    """Nạp mọi lib của ``package`` vào một module mới ``<package>.ws`` rồi chạy các ``stages`` theo thứ tự yêu cầu.

    gpu=False bỏ qua các lib cần torch/GPU (đủ cho phần phân tích + vẽ hình của Study 1 và Study 2a).
    Module trả về có thêm ``run_stage(name)`` để chạy tiếp stage khác trên cùng state.
    """
    pkg_dir = Path(sys.modules[package].__file__).parent
    by_name = {s.name: s for s in steps if s.kind == "stage"}
    unknown = [s for s in stages if s not in by_name]
    if unknown:
        raise ValueError(f"stage không tồn tại: {unknown}. Có: {list(by_name)}")
    if not gpu:
        need_gpu = [s for s in stages if by_name[s].gpu]
        if need_gpu:
            raise ValueError(f"stage {need_gpu} cần GPU/torch nhưng gpu=False")

    ws = types.ModuleType(f"{package}.ws", f"Workspace của {package} (namespace chung như kernel notebook).")
    sys.modules[ws.__name__] = ws  # để pickle / multiprocessing tìm được hàm định nghĩa trong workspace
    get = Params(params)
    ns = ws.__dict__
    ns["__builtins__"] = builtins
    ns["param"] = get

    def _exec(step: Step):
        path = pkg_dir / step.file
        t0 = time.time()
        if verbose:
            print(
                f"[hourglass] {step.kind:5s} {step.name:22s} <- {path.relative_to(pkg_dir.parent.parent.parent)}",
                flush=True,
            )
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
        if verbose and time.time() - t0 > 5:
            print(f"[hourglass]       {step.name}: {time.time() - t0:.0f}s", flush=True)

    for step in steps:
        if step.kind == "lib" and (gpu or not step.gpu):
            _exec(step)

    unused = set(params) - get.used - GLOBAL_KEYS
    if unused:
        warnings.warn(f"[hourglass] tham số không được study {package} dùng (sai tên?): {sorted(unused)}", stacklevel=2)

    def run_stage(name: str):
        if name not in by_name:
            raise ValueError(f"stage không tồn tại: {name}. Có: {list(by_name)}")
        if by_name[name].gpu and not gpu:
            raise ValueError(f"stage {name} cần GPU/torch nhưng workspace được nạp với gpu=False")
        _exec(by_name[name])

    ns["run_stage"] = run_stage
    ns["__steps__"] = steps
    for name in stages:
        run_stage(name)
    return ws
