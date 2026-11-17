"""fusion degrades to the unfused path when torch lacks the private
inductor names it patches (torch 2.15 has no disable_pointwise_autotuning)."""
import warnings

import pytest

torch = pytest.importorskip('torch')
from torchvic.core import fusion                           # noqa: E402


def test_missing_private_api_turns_fusion_off_with_one_warning(monkeypatch):
    th = pytest.importorskip('torch._inductor.runtime.triton_heuristics')
    monkeypatch.setattr(fusion, '_api_checked', None)
    monkeypatch.setattr(fusion, '_ready', False)
    monkeypatch.setattr(fusion, 'FUSION', True)
    monkeypatch.delattr(th, 'disable_pointwise_autotuning', raising=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fusion.setup()                  # must not raise
        fusion.setup()
        assert fusion.api_available() is False
    msgs = [x for x in w if 'lacks inductor internals' in str(x.message)]
    assert len(msgs) == 1
    assert 'disable_pointwise_autotuning' in str(msgs[0].message)
    assert fusion.FUSION is False
    assert fusion._ready is False
    assert fusion.wants([torch.zeros(1)]) is False


def test_present_private_api_is_accepted(monkeypatch):
    pytest.importorskip('torch._inductor.runtime.triton_heuristics')
    monkeypatch.setattr(fusion, '_api_checked', None)
    monkeypatch.setattr(fusion, 'FUSION', True)
    ok = fusion.api_available()
    assert ok is (fusion.FUSION is True)
