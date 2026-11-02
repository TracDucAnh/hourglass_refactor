#!/usr/bin/env bash
# Study 2a: chạy các lưới theo thứ tự cổng của kế hoạch (G0 -> pilot -> G2 -> lưới chính). Mọi lưới có resume.
#
#   scripts/sweep.sh g0       # testcase bộ sinh, không cần GPU
#   scripts/sweep.sh smoke    # đường ống cực nhỏ, vài phút
#   scripts/sweep.sh pilot    # lưới pilot_conv + báo cáo -> chốt BATCH / SEQ_MULT trước khi chạy tiếp
#   scripts/sweep.sh gate     # cổng G2 + báo cáo + dự toán GPU-giờ
#   scripts/sweep.sh grids    # toàn bộ lưới chính rồi phân tích
#   scripts/sweep.sh all      # MAIN RESULTS trọn gói: gate_g2 + 7 lưới chính (450 run) + phân tích; cổng chỉ là báo cáo,
#                             # KHÔNG chặn lưới. Chạy lưới 2 lượt: lượt 2 chỉ thử lại run bị lỗi (run `ok` được bỏ qua).
#
# Tham số thêm được chuyển cho train.py, vd: scripts/sweep.sh gate -o BATCH=512 -o SEQ_MULT=4
set -euo pipefail
cd "$(dirname "$0")/.."
PY="uv run python"
step="${1:-}"; shift || true

case "$step" in
  g0)    $PY -m hourglass run study2a g0_generator_tests --no-gpu "$@" ;;
  smoke) $PY scripts/train.py study2a_smoke gate_report "$@"
         $PY scripts/make_figures.py study2a_smoke "$@" ;;
  pilot) $PY scripts/train.py study2a_pilot_conv pilot_report "$@" ;;
  gate)  $PY scripts/train.py study2a_gate_g2 gate_report "$@" ;;
  grids) $PY scripts/train.py study2a_grids status "$@"
         $PY scripts/make_figures.py study2a "$@" ;;
  all)   $PY scripts/train.py study2a_gate_g2 gate_report "$@"    # báo cáo G2 (không chặn)
         $PY scripts/train.py study2a_grids "$@"                  # lượt 1: p1p2 balanced moe_diff p3s2 s4 s5 heldout_p4
         $PY scripts/train.py study2a_grids status "$@"           # lượt 2: chỉ thử lại run lỗi (vd. OOM)
         $PY scripts/make_figures.py study2a "$@" ;;
  *) grep '^#' "$0" | sed -n '2,15p'; exit 1 ;;
esac
