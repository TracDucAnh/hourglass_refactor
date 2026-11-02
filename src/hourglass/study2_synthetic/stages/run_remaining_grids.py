# Chạy các lưới còn lại theo thứ tự gợi ý (Cell C6)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 33 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["run_remaining_grids"]) hoặc `python -m hourglass run <experiment> run_remaining_grids`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell C6 — CHẠY các lưới còn lại (mỗi lưới một lần gọi; có resume nên chạy lại bao nhiêu lần cũng được)
# Bỏ comment từng dòng khi cần. Thứ tự gợi ý theo giá trị khoa học / chi phí.
# ============================================================
run_grid("p1p2")        # 108 run: P1 (decoder, encoder), P2
run_grid("balanced")    #  24 run: cánh cân bằng giao diện (A1)
run_grid("moe_diff")    #  36 run: MoE + diffusion (P1, S6)
run_grid("p3s2")        # 180 run (+72 sinh đôi): P3, S2, S3
run_grid("s4")          #  24 run: S4 (pivot)
run_grid("s5")          #  30 run: S5
run_grid("heldout_p4")  #  48 run: giữ ra cho P4 (đầu vào của Study 5)
show_status()
