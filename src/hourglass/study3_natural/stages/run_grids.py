# Chạy các lưới trong RUN_GRIDS, có resume; FAMILY / SHARD để chia cho nhiều GPU (Study 2b)
# STAGE (hành động): chạy bằng load(stages=["run_grids"]) hoặc `python -m hourglass run <experiment> run_grids`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study3_natural` (một namespace chung): KHÔNG import trực tiếp.

for _g in RUN_GRIDS:
    run_grid(_g)
