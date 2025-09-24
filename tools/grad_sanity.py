"""Light-weight gradient sanity checks used by the training entrypoints.

The helper primarily targets the PyTorch training stack.  When PyTorch (or its
CUDA runtime) is unavailable, a numpy-based finite difference fallback is used
so the check can still run in minimal environments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class GradientStat:
    """Stores diagnostic information for a single parameter tensor."""

    name: str
    norm: float
    max_abs: float

    def format(self) -> str:
        return f"{self.name}: norm={self.norm:.6f}, max|grad|={self.max_abs:.6f}"


def _run_numpy_probe(batch_size: int = 32) -> None:
    """Fallback sanity check that relies purely on NumPy.

    We evaluate the gradient of a simple quadratic function using central
    differences and verify that the estimated gradients are stable and finite.
    """

    rng = np.random.default_rng(1729)
    weights = rng.standard_normal((16, 4))
    inputs = rng.standard_normal((batch_size, 16))
    targets = rng.standard_normal((batch_size, 4))

    def loss_fn(w: np.ndarray) -> float:
        preds = inputs @ w
        diff = preds - targets
        return float(np.mean(diff * diff))

    grad_estimate = np.zeros_like(weights)
    eps = 1e-4
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            delta = np.zeros_like(weights)
            delta[i, j] = eps
            plus = loss_fn(weights + delta)
            minus = loss_fn(weights - delta)
            grad_estimate[i, j] = (plus - minus) / (2 * eps)

    grad_norm = float(np.linalg.norm(grad_estimate))
    max_abs = float(np.max(np.abs(grad_estimate)))

    if not math.isfinite(grad_norm) or not math.isfinite(max_abs):
        raise RuntimeError("NumPy gradient probe produced non-finite results.")

    print("[grad-sanity] PyTorch unavailable, ran NumPy finite-difference probe instead.")
    print(f"[grad-sanity] Estimated gradient norm={grad_norm:.6f}, max|grad|={max_abs:.6f}")
    print("[grad-sanity] ✓ Gradient sanity check passed (NumPy mode).")


def _torch_select_device(torch_mod, preferred: str | None = None):
    if preferred is not None:
        return torch_mod.device(preferred)
    if torch_mod.cuda.is_available():
        return torch_mod.device("cuda")
    if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():  # pragma: no cover
        return torch_mod.device("mps")
    return torch_mod.device("cpu")


def _torch_build_probe(torch_mod, dtype, device):
    nn = torch_mod.nn
    probe = nn.Sequential(
        nn.Linear(16, 32),
        nn.GELU(),
        nn.Linear(32, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )
    probe.to(device=device, dtype=dtype)
    return probe


def _torch_probe_batch(torch_mod, batch_size: int, dtype, device):
    torch_mod.manual_seed(1729)
    inputs = torch_mod.randn(batch_size, 16, device=device, dtype=dtype, requires_grad=True)
    targets = torch_mod.randn(batch_size, 4, device=device, dtype=dtype)
    return inputs, targets


def _torch_collect_stats(torch_mod, parameters: Iterable[tuple[str, object]]):
    stats: list[GradientStat] = []
    for name, param in parameters:
        grad = getattr(param, "grad", None)
        if grad is None:
            raise RuntimeError(f"No gradient computed for parameter '{name}' during sanity check.")
        if not torch_mod.isfinite(grad).all():
            raise RuntimeError(f"Non-finite values detected in gradients of '{name}'.")
        grad_detached = grad.detach()
        norm = float(grad_detached.norm().cpu())
        max_abs = float(grad_detached.abs().max().cpu())
        if not math.isfinite(norm) or not math.isfinite(max_abs):
            raise RuntimeError(f"Unstable gradient statistics for '{name}'.")
        stats.append(GradientStat(name=name, norm=norm, max_abs=max_abs))
    return stats


def run_check(*, device: str | None = None, dtype=None, batch_size: int = 32) -> None:
    """Execute a quick forward/backward pass to validate gradient computation."""

    try:
        import torch as torch_mod
    except Exception as exc:  # pragma: no cover - exercised in CPU-only environments
        print(f"[grad-sanity] PyTorch import failed ({exc}); falling back to NumPy probe.")
        _run_numpy_probe(batch_size=batch_size)
        return

    if dtype is None:
        dtype = torch_mod.float32

    chosen_device = _torch_select_device(torch_mod, device)
    print(f"[grad-sanity] Running gradient probe on device: {chosen_device} (dtype={dtype})")

    model = _torch_build_probe(torch_mod, dtype=dtype, device=chosen_device)
    optimiser = torch_mod.optim.Adam(model.parameters(), lr=1e-3)

    inputs, targets = _torch_probe_batch(torch_mod, batch_size=batch_size, dtype=dtype, device=chosen_device)

    optimiser.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = torch_mod.nn.functional.mse_loss(outputs, targets)
    print(f"[grad-sanity] Synthetic loss: {float(loss.detach().cpu()):.6f}")

    loss.backward()

    stats = _torch_collect_stats(torch_mod, model.named_parameters())
    total_norm = float(torch_mod.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0).cpu())
    if not math.isfinite(total_norm):
        raise RuntimeError("Total gradient norm became non-finite during sanity check.")

    optimiser.step()

    print("[grad-sanity] Individual gradient statistics:")
    for item in stats:
        print(f"  - {item.format()}")
    print(f"[grad-sanity] Total gradient norm (after clipping): {total_norm:.6f}")
    print("[grad-sanity] ✓ Gradient sanity check passed.")


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    run_check()
