# Định nghĩa lưới + run_id (Cell C1)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 26 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell C1 — Định nghĩa lưới + run_id
# ============================================================
FAMILIES = ["decoder", "encoder", "moe", "diffusion"]
TASKS = ["LM", "SEM", "SURF"]

def make_cfg(grid, family="decoder", level="M4", task="SEM", d=D_MAIN, L=4, overlap=0, len_scale=1.0,
             len_scale_X=None, share_X=None, seed=0, bs=None, seq_mult=None, tag=None):
    """bs / seq_mult / tag: chỉ dùng cho lưới pilot_conv (ghi đè batch và hệ số nhân tổng chuỗi cho riêng run đó)."""
    return dict(grid=grid, family=family, level=level, task=task, d=d, L=L, overlap=overlap, len_scale=len_scale,
                len_scale_X=len_scale_X, share_X=share_X, seed=seed, bs=bs, seq_mult=seq_mult, tag=tag)

def run_id(c):
    ls = f"{c['len_scale']:g}" + (f"X{c['len_scale_X']:g}" if c["len_scale_X"] else "")
    sh = f"{c['share_X']:.2f}" if c["share_X"] else "eq"
    return (f"{c['grid']}__{c['family']}_{c['level']}_{c['task']}_d{c['d']}_L{c['L']}_ov{c['overlap']}_ls{ls}_sh{sh}_s{c['seed']}"
            + (f"_{c['tag']}" if c.get("tag") else ""))

def grid_configs(name):
    if name == "pilot_conv":
        # Cùng cấu hình với cổng G2 (decoder M4/SEM, d=192, L=4), 3 seed. Hai câu hỏi:
        #  (i) cùng TỔNG CHUỖI, nhiều lần cập nhật hơn có giúp không (b1024x1 vs b512x1)?
        #  (ii) thêm chuỗi (x2, x4) còn cải thiện val / dịch k* không? -> chọn ngân sách nhỏ nhất đã ổn định (Cell C5b).
        arms = ([("b64x1", 64, 1.0), ("b32x1", 32, 1.0), ("b64x2", 64, 2.0)] if SMOKE else
                [("b1024x1", 1024, 1.0), ("b512x1", 512, 1.0), ("b1024x2", 1024, 2.0), ("b1024x4", 1024, 4.0)])
        return [make_cfg("pilot_conv", seed=s, bs=bs, seq_mult=m, tag=tag) for tag, bs, m in arms for s in range(3)]
    if name == "gate_g2":
        return [make_cfg("gate_g2", seed=s) for s in range(3)]
    if name == "smoke":
        out = [make_cfg("smoke", family=f, d=64, L=3, task="SEM") for f in FAMILIES]
        return out + [make_cfg("smoke", family="decoder", d=64, L=3, task="LM", overlap=100, level="M2")]
    if name == "p1p2":
        return [make_cfg("p1p2", family=f, level=lv, task=t, seed=s)
                for f in ("decoder", "encoder") for lv in LEVELS for t in TASKS for s in range(6)]
    if name == "moe_diff":
        return [make_cfg("moe_diff", family=f, task=t, seed=s) for f in ("moe", "diffusion") for t in TASKS for s in range(6)]
    raise KeyError(name)

ALL_GRIDS = ["pilot_conv", "gate_g2", "smoke", "p1p2", "moe_diff"]
print({g: len(grid_configs(g)) for g in ALL_GRIDS})
