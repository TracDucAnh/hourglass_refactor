"""Cổng G0 của Study 2a (kế hoạch mục 2): testcase bộ sinh ngôn ngữ tổng hợp + số đo, không cần GPU."""

import pytest


@pytest.mark.slow
def test_g0_generator_gate():
    from hourglass.study2_synthetic import load

    ws = load("study2a", gpu=False, stages=["g0_generator_tests"], verbose=False)  # stage tự assert khi có FAIL
    failed = [name for name, ok, _ in ws.TEST_LOG if not ok]
    assert len(ws.TEST_LOG) >= 40 and not failed, failed
