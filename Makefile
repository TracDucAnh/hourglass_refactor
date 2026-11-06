# Lối tắt cho các bước chính. `make help` để xem danh sách. Mọi lệnh chạy trong môi trường uv (uv.lock).
PY := uv run python
O  ?=            # override tham số, vd: make study1 O='-o RUN_MODELS=[xlmr_base]'

.PHONY: help install install-core lint format test test-fast g0 \
        study1-g1 study1 study2a-smoke study2a-pilot study2a-gate study2a-grids \
        study2a-all study2a-decoder study2a-encoder study2a-moe study2a-diffusion study2a-report study4-prepare study4-pilot study4 figures clean

help:           ## Liệt kê lệnh
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:        ## Cài đủ (kể cả extra study4) + pre-commit
	uv sync --all-extras
	uv run pre-commit install

install-core:   ## Cài tối thiểu cho Study 1 và 2a
	uv sync

lint:           ## ruff check (không lint mã study giữ nguyên văn từ notebook)
	uv run ruff check src scripts tests

format:         ## ruff format
	uv run ruff format src scripts tests

test:           ## Toàn bộ test (kể cả G0 bộ sinh, ~15 s)
	uv run pytest

test-fast:      ## Test nhanh (bỏ slow)
	uv run pytest -m "not slow"

g0:             ## Cổng G0 Study 2a: testcase bộ sinh + số đo (CPU)
	$(PY) -m hourglass run study2a g0_generator_tests --no-gpu $(O)

# ---------------------------------------------------------------- Study 1 (suy luận, GPU)
study1-g1:      ## Cổng G1: XLM-R base, Qwen3-1.7B, OLMoE trên FLORES
	$(PY) scripts/evaluate.py study1_gate_g1 $(O)

study1:         ## Study 1 đầy đủ: 15 model + random-init, FLORES + NTREX
	$(PY) scripts/evaluate.py study1 $(O)

# ---------------------------------------------------------------- Study 2a (huấn luyện, GPU)
study2a-smoke:  ## Đường ống cực nhỏ (vài phút; CPU được)
	scripts/sweep.sh smoke $(O)

study2a-pilot:  ## Lưới pilot hội tụ -> chốt BATCH/SEQ_MULT
	scripts/sweep.sh pilot $(O)

study2a-gate:   ## Cổng G2 + dự toán GPU-giờ
	scripts/sweep.sh gate $(O)

study2a-grids:  ## Toàn bộ lưới chính + phân tích
	scripts/sweep.sh grids $(O)

study2a-all:    ## MAIN RESULTS 2a trọn gói: G2 (chỉ báo cáo) + 216 run (p1p2, moe_diff, p2_seeds + chẩn đoán điểm neo) + thử lại run lỗi + phân tích
	scripts/sweep.sh all $(O)

study2a-decoder:   ## Chỉ họ decoder (gate_g2, p1p2, p2_seeds, mask_ablation, anchors); chia GPU: O='-o SHARD=0/2'
	$(PY) scripts/train.py study2a_decoder status $(O)

study2a-encoder:   ## Chỉ họ encoder (p1p2, p2_seeds, mask_ablation); chia GPU: O='-o SHARD=0/2'
	$(PY) scripts/train.py study2a_encoder status $(O)

study2a-moe:       ## Chỉ họ MoE (moe_diff, p2_seeds)
	$(PY) scripts/train.py study2a_moe status $(O)

study2a-diffusion: ## Chỉ họ diffusion (moe_diff, p2_seeds)
	$(PY) scripts/train.py study2a_diffusion status $(O)

study2a-report:    ## Phân tích + hình + báo cáo Study 2a (sau khi mọi họ chạy xong)
	$(PY) scripts/make_figures.py study2a $(O)

# ---------------------------------------------------------------- Study 4 (cần extra study4)
study4-prepare: ## Dữ liệu Study 4: tách từ, tần suất FineWeb-2, Eflomal, ITEMS (CPU, lâu, có cache)
	$(PY) -m hourglass run study4 check_data --no-gpu $(O)

study4-pilot:   ## Truy hồi mức từ cho XLM-R large (kiểm tra trước)
	$(PY) scripts/evaluate.py study4_pilot $(O)

study4:         ## Truy hồi mức từ cho mọi model
	$(PY) scripts/evaluate.py study4 $(O)

# ---------------------------------------------------------------- Bảng + hình
figures:        ## Sinh lại bảng (results/) + hình (paper/figures/) cho cả ba study
	$(PY) scripts/make_figures.py $(O)

clean:          ## Xoá cache Python / pytest / ruff (KHÔNG xoá data/, outputs/, results/)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
