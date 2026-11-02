"""Mọi test ghi dữ liệu / kết quả vào thư mục tạm, không đụng data/, outputs/, results/ của repo."""

import os
import tempfile

os.environ["HOURGLASS_ROOT"] = tempfile.mkdtemp(prefix="hourglass-test-")
os.environ["MPLBACKEND"] = "Agg"
