# Train tokenizer cần cho RUN_GRIDS rồi mã hoá corpus (Study 2b)
# STAGE (hành động): chạy bằng load(stages=["build_tokens"]) hoặc `python -m hourglass run <experiment> build_tokens`.
# File này là một phần của workspace `hourglass.study3_natural` (một namespace chung): KHÔNG import trực tiếp.

build_tokens(token_needs(RUN_GRIDS))
