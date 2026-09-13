"""Checkpoint loading: memory-map the multi-GB file, but never at the cost of
breaking loads that cannot be mapped. Runs without torch installed by injecting
a stand-in module (this dev box is CPU-only)."""
import sys
import types

import pytest

from src.roofs.sam3_predictors import _load_checkpoint


def _fake_torch(monkeypatch, behaviour):
    """Install a stand-in torch whose load() records kwargs and applies behaviour."""
    calls = []

    def load(path, **kw):
        calls.append(kw)
        return behaviour(kw)

    mod = types.ModuleType("torch")
    mod.load = load
    monkeypatch.setitem(sys.modules, "torch", mod)
    return calls


def test_checkpoint_is_memory_mapped_when_supported(monkeypatch):
    calls = _fake_torch(monkeypatch, lambda kw: {"model": {}, "epoch": 6})
    out = _load_checkpoint("/tmp/ckpt.pt")
    assert out["epoch"] == 6
    assert len(calls) == 1 and calls[0]["mmap"] is True      # mapped, not slurped


@pytest.mark.parametrize("exc", [TypeError, ValueError, RuntimeError])
def test_falls_back_when_mmap_unavailable(monkeypatch, exc):
    """torch<2.1 has no mmap kwarg; a non-zipfile checkpoint can't be mapped.
    Either way we must still load it, not raise — this is exactly what a global
    torch.load monkeypatch forcing mmap gets wrong."""
    def behaviour(kw):
        if kw.get("mmap"):
            raise exc("no mmap here")
        return {"model": {}, "epoch": 9}

    calls = _fake_torch(monkeypatch, behaviour)
    out = _load_checkpoint("/tmp/ckpt.pt")
    assert out["epoch"] == 9                                 # loaded anyway
    assert len(calls) == 2                                   # tried mmap, then fell back
    assert calls[0]["mmap"] is True and "mmap" not in calls[1]
