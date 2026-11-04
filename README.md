# hourglass — Principle of Least Effort in Depth

Mã thí nghiệm cho bài **"Principle of Least Effort in Depth: Explaining the Diverge–Converge–Diverge Hourglass in
Multilingual Language Models"** (Anh Trac Duc Dinh). Mô hình đa ngữ tổ chức biểu diễn theo ba pha theo độ sâu — tách
theo ngôn ngữ ở tầng đầu, hội tụ ở tầng giữa, tách lại gần đầu ra (DCD). Nguyên lý LE-D giải thích hình đồng hồ cát
này bằng một mục tiêu information bottleneck hai số hạng theo từng tầng; repo kiểm định các giả thuyết P1–P4, S1–S6
rút ra từ đó.

| Study | Câu hỏi | Cách làm | Package |
|---|---|---|---|
| **1** | Hourglass có thật ở cả 4 họ (encoder, decoder, MoE, diffusion)? (P1, S6) | suy luận trên 15 mô hình mở + bản random-init, FLORES-200 / NTREX-128, số đo M1–M4 | `hourglass.study1_open_models` |
| **2a** | Eo có theo đúng thứ các tầng sau cần? (P1, P2, P3, S2–S6) | ngôn ngữ tổng hợp từ một PCFG ngữ nghĩa + mô hình tí hon K = 8, đổi từng nhân tố một | `hourglass.study2_synthetic` |
| **4** | Từ nào, ngôn ngữ nào được chia sẻ ở eo? (S3) | truy hồi mức từ theo log-hạng tần suất (GAM, bootstrap), Mantel với khoảng cách loại hình | `hourglass.study4_frequency` |

Bản đồ đầy đủ giả thuyết ↔ study ↔ mã ↔ config ↔ kết quả, và các cổng G0–G3: [docs/experiments.md](docs/experiments.md).

## Cài đặt

Cần [uv](https://docs.astral.sh/uv/) và Python 3.10–3.12.

```bash
uv sync                    # Study 1, 2a (torch 2.8, transformers 4.57.1 — theo pin của notebook)
uv sync --all-extras       # + Study 4 (jieba, fugashi, pythainlp, stanza, eflomal, lang2vec, ...)
uv run pre-commit install  # hook: ruff, nbstripout, chặn file lớn / khoá bí mật
export HF_TOKEN=...        # FLORES-200 và một số model là gated trên Hugging Face
```

`uv.lock` khoá phiên bản để tái lập môi trường. Mặc định dữ liệu và kết quả nằm trong repo (`data/`, `outputs/`,
`results/`, `paper/figures/`); đặt `HOURGLASS_ROOT=/đường/dẫn` để chuyển sang ổ khác.

## Chạy lại kết quả chính

```bash
make help              # danh sách lệnh
make test              # parity với notebook + config + G0 + TinyLM (~15 s, CPU)

# Study 1 (GPU 80 GB, bf16)
make study1-g1         # cổng G1: XLM-R base, Qwen3-1.7B, OLMoE trên FLORES (~3 GPU-giờ)
make study1            # 15 model + random-init, FLORES + NTREX (~20 GPU-giờ)

# Study 2a (GPU)
make g0                # cổng G0: testcase bộ sinh + số đo (CPU, ~10 s)
make study2a-smoke     # cả đường ống ở cỡ cực nhỏ (CPU được, vài phút)
make study2a-pilot     # chốt BATCH / SEQ_MULT  ->  make study2a-gate (G2)  ->  make study2a-grids
make study2a-decoder   # hoặc chạy từng họ song song (mỗi họ một GPU, xem docs/studies/study2a.md):
make study2a-encoder   #   study2a-{decoder,encoder,moe,diffusion}, rồi make study2a-report
make study2a-moe
make study2a-diffusion
make study2a-report

# Study 4 (extra study4; bước chuẩn bị chạy lâu nhưng có cache)
make study4-prepare    # tách từ + UPOS, đếm FineWeb-2, Eflomal + MUSE/Wiktionary -> ITEMS
make study4-pilot      # XLM-R large trước, rồi: make study4

make figures           # bảng -> results/<study>/, hình -> paper/figures/<study>/ (không chạy lại mô hình)
```

Mọi bước có resume (run đã có kết quả được bỏ qua). Đổi tham số không cần sửa file: thêm override, ví dụ
`make study1 O='-o RUN_MODELS=[qwen3_8b] -o FORCE_RERUN=true'` hoặc
`uv run python -m hourglass run study4 s3 mantel -o N_BOOT=200`.

## Cấu trúc

```
├── configs/                  mọi siêu tham số (YAML), ghép bằng hourglass.config
│   ├── config.yaml           đường dẫn gốc, SEED
│   ├── data/                 flores_ntrex12, synthetic (2a), lexical (4)
│   ├── model/                tiny_lm (2a)            ← registry model mở: src/.../registry.py
│   ├── trainer/              synthetic_v2 (giao thức huấn luyện v2 của 2a)
│   └── experiment/           mỗi file = 1 thí nghiệm / cổng: study1, study1_gate_g1, study2a_gate_g2, study4, ...
├── src/hourglass/
│   ├── config.py, workspace.py, cli.py     hạ tầng: ghép config, nạp workspace, dòng lệnh
│   ├── common/               hàm giống hệt nhau giữa các notebook: io, text, net, hf, plotting
│   ├── study1_open_models/   config, registry, data, models, extract, metrics, runner, analysis, plotting/, stages/
│   ├── study2_synthetic/     config, synth/ (PCFG, renderer, parser, BPE, world), models, training, extract,
│   │                         metrics, logit_lens, grids, runner, gate, pilot, analysis/, plotting/, report, stages/
│   └── study4_frequency/     config, registry, data, words, frequency, alignment, lemmas, items, extract,
│                             retrieval, runner, tables, analysis/, plotting/, diagnostics/, stages/
├── scripts/                  entry point: train.py (2a), evaluate.py (1, 4), make_figures.py, sweep.sh
├── notebooks/                khám phá / vẽ (01–03); archive/ = notebook gốc, nguồn của toàn bộ mã
├── tests/                    parity với notebook, config, G0, TinyLM (shape, overfit 1 batch)
├── data/  outputs/           dữ liệu + kết quả thô (gitignore)
├── results/                  bảng tổng hợp nhỏ (commit được)
├── paper/figures/            hình PDF
└── docs/                     bản đồ thí nghiệm, kiến trúc, vấn đề đã biết, ghi chú từng study (PDF không commit)
```

### Mã của từng study: workspace thay vì import

Notebook gốc chạy trong một kernel, hàm ở cell sau đọc biến toàn cục do cell trước tạo (vd. `ITEMS`, `WT` ở
Study 4). Để **không đổi logic**, mỗi cell được chuyển nguyên văn thành một file theo chức năng, và các file của một
study được nạp vào một namespace chung theo đúng thứ tự cell:

```python
from hourglass.study1_open_models import load
ws = load("study1", gpu=False)        # lib phân tích + vẽ, không cần GPU
df = ws.feature_table("flores")
ws.run_stage("analyze_flores")        # stage = một cell hành động của notebook
```

`python -m hourglass stages <experiment>` liệt kê lib / stage; header mỗi file ghi cell gốc. `tests/test_notebook_parity.py`
kiểm (theo AST) rằng mã trong `src/` tương đương notebook trong `notebooks/archive/`, chỉ khác ở `param(...)` đọc
config và các hàm dùng chung chuyển vào `common/`. Chi tiết: [docs/architecture.md](docs/architecture.md).

## Lưu ý

- [docs/known_issues.md](docs/known_issues.md) ghi các vấn đề phát hiện khi tổ chức lại, **chưa sửa** vì giữ
  nguyên logic — quan trọng nhất: Study 1 báo lỗi `UnboundLocalError: JS` ở mọi run mới của model không phải MoE.
- Mọi kết luận chỉ "ở quy mô nhỏ"; Study 2b/2c, 3 và 5 của kế hoạch chưa có trong repo.
