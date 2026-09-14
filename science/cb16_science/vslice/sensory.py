"""R12 VS-B frozen sensory boundary.

The frozen sensory module maps the canonical normalized market tensor
``[B, L, 5]`` (or one ``[L, 5]`` sample) to the deterministic representation
``Z_t`` of width ``z_dim``:

```text
Z_t = tanh(flatten(normalized_market) @ P^T),  P in R^{z_dim x (L*5)}
```

``P`` is a fixed random projection drawn from the local fixed seed
``12012`` and scaled by ``1/sqrt(L*5)`` so the projected variance stays near
one.  It is registered as a module buffer, so the module exposes zero trainable
parameters and can never be reached by an optimizer.

The fixed random projection is a controlled placeholder for the frozen sensory
role of the vertical slice.  It is not a pretrained market model and no
learnability claim is attached to it.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .contracts import CONTEXT_LENGTH, ContractError

#: Local deterministic seed for the frozen projection buffer.
SENSORY_SEED = 12012
#: Frozen representation width for VS-B.
Z_DIM = 32
#: Canonical number of normalized market channels (pO, pH, pL, pC, v).
MARKET_CHANNELS = 5


def _require_market_tensor(market: Any) -> torch.Tensor:
    """Fail closed on anything that is not a finite floating market tensor."""

    if not isinstance(market, torch.Tensor):
        raise ContractError(f"market must be a torch.Tensor, got {type(market).__name__}")
    if market.ndim not in (2, 3):
        raise ContractError(
            "market must have shape [L, 5] or [B, L, 5], got shape "
            f"{tuple(market.shape)}"
        )
    if market.shape[-1] != MARKET_CHANNELS:
        raise ContractError(
            f"market must carry exactly {MARKET_CHANNELS} normalized channels, got "
            f"{market.shape[-1]}"
        )
    if not market.dtype.is_floating_point:
        raise ContractError(f"market must be a floating tensor, got dtype {market.dtype}")
    if not bool(torch.isfinite(market).all()):
        raise ContractError("market tensor must be finite")
    return market


class FrozenSensory(nn.Module):
    """Deterministic frozen projection from normalized market state to ``Z_t``.

    Input is one canonical normalized market tensor of shape ``[L, 5]`` or a
    batch ``[B, L, 5]``.  Output is ``[z_dim]`` for a single sample and
    ``[B, z_dim]`` for a batch.  The module is pure and deterministic: identical
    inputs produce bitwise-identical outputs, and it holds no trainable
    parameter.

    The canonical normalized tensor may arrive as ``float64`` (the VS-A
    normalizer returns ``float64`` NumPy), so the flattened input is cast to the
    frozen projection dtype at this boundary; the module's own dtype defines the
    representation precision.
    """

    def __init__(
        self,
        *,
        z_dim: int = Z_DIM,
        context_length: int = CONTEXT_LENGTH,
        seed: int = SENSORY_SEED,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if isinstance(z_dim, bool) or not isinstance(z_dim, int) or z_dim < 1:
            raise ContractError(f"z_dim must be a positive int, got {z_dim!r}")
        if (
            isinstance(context_length, bool)
            or not isinstance(context_length, int)
            or context_length < 1
        ):
            raise ContractError(f"context_length must be a positive int, got {context_length!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ContractError(f"seed must be an int, got {seed!r}")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ContractError(f"dtype must be a floating torch dtype, got {dtype!r}")

        self.z_dim = z_dim
        self.context_length = context_length
        self.seed = seed
        self.dtype = dtype

        generator = torch.Generator(device="cpu").manual_seed(seed)
        flat_dim = context_length * MARKET_CHANNELS
        projection = torch.randn(z_dim, flat_dim, generator=generator, dtype=dtype)
        projection = projection * (1.0 / math.sqrt(flat_dim))
        # A buffer is never returned by ``parameters()`` and never receives a
        # gradient, which is exactly the frozen-sensory contract.
        self.register_buffer("projection", projection)

    @property
    def flat_dim(self) -> int:
        """Flattened normalized-market width the projection consumes."""

        return self.context_length * MARKET_CHANNELS

    def forward(self, market: torch.Tensor) -> torch.Tensor:
        tensor = _require_market_tensor(market)
        if tensor.shape[-2] != self.context_length:
            raise ContractError(
                f"market must carry exactly context_length={self.context_length} represented "
                f"bars, got {tensor.shape[-2]}"
            )
        single = tensor.ndim == 2
        batch = tensor.unsqueeze(0) if single else tensor
        if batch.shape[0] < 1:
            raise ContractError("market batch must hold at least one sample")
        flat = batch.reshape(batch.shape[0], self.flat_dim)
        cast_flat = flat.to(self.projection.dtype)
        # A finite float64 value beyond the projection dtype's range would cast
        # to infinity; fail here instead of letting tanh silently saturate.
        if not bool(torch.isfinite(cast_flat).all()):
            raise ContractError(
                f"market tensor must stay finite in the projection dtype {self.projection.dtype}"
            )
        projected = torch.matmul(cast_flat, self.projection.t())
        z = torch.tanh(projected)
        if not bool(torch.isfinite(z).all()):
            raise ContractError("frozen sensory output must be finite")
        return z[0] if single else z


__all__ = ["MARKET_CHANNELS", "SENSORY_SEED", "Z_DIM", "FrozenSensory"]
