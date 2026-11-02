# Tiện ích kiểm thử bộ sinh (Cell T0)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 12 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell T0 — Tiện ích kiểm thử
# ============================================================
TEST_LOG = []
def check(name, ok, detail=""):
    TEST_LOG.append((name, bool(ok), str(detail)))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail != "" else ""))

def walk_nps(S, depth=1):
    """Duyệt mọi NP của một mệnh đề, trả về (np, độ sâu)."""
    for a in S[5]:
        yield from _walk_np(a, depth)

def _walk_np(np_, depth):
    yield np_, depth
    if np_[5] is not None:
        if np_[5][0] == "POSS":
            yield from _walk_np(np_[5][1], depth + 1)
        else:
            yield from walk_nps(np_[5][1], depth + 1)

def walk_clauses(S, depth=1):
    yield S, depth
    for np_, d in walk_nps(S, depth):
        if np_[5] is not None and np_[5][0] == "REL":
            yield np_[5][1], d + 1

def _z(p_hat, p, n):
    return abs(p_hat - p) / math.sqrt(max(p * (1 - p), 1e-12) / n)
