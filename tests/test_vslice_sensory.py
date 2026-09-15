"""Frozen-sensory regression tests: required tests 1-5 of the R12 VS-B contract.

Required by ``docs/experiments/frozen/R12_VS_B_LEARNER_SPINE_R0.md`` (section "Required
tests", "Frozen sensory").  Test 4 (the projection buffer survives a valid
Actor/Critic update) needs a learner and lives in ``test_vslice_learner.py``.

    .venv/bin/python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:  # CPU PyTorch lives in the task venv; a bare interpreter records a skip.
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without PyTorch
    raise unittest.SkipTest(f"CPU PyTorch is required for the R12 VS-B learner spine: {exc}")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "science") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "science"))

from cb16_science.vslice.contracts import CONTEXT_LENGTH, ContractError  # noqa: E402
from cb16_science.vslice.market import normalize_market_window  # noqa: E402
from cb16_science.vslice.sensory import (  # noqa: E402
    MARKET_CHANNELS,
    SENSORY_SEED,
    Z_DIM,
    FrozenSensory,
)


def market_window(seed: int = 12012, context_length: int = CONTEXT_LENGTH) -> torch.Tensor:
    """A deterministic finite ``[L, 5]`` stand-in for a normalized market tensor."""

    generator = torch.Generator().manual_seed(seed)
    return torch.randn(context_length, MARKET_CHANNELS, generator=generator) * 0.05


def canonical_raw_window(context_length: int = CONTEXT_LENGTH) -> np.ndarray:
    """One retained predecessor plus ``L`` correctly ordered OHLCV bars."""

    close = 100.0 * np.exp(0.05 * np.linspace(0.0, 1.0, context_length + 1))
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = np.full(context_length + 1, 10.0)
    return np.column_stack([open_, high, low, close, volume])


class FrozenSensoryTests(unittest.TestCase):
    def test_01_same_input_is_bitwise_identical_on_cpu(self):
        sensory = FrozenSensory()
        market = market_window()

        first = sensory(market)
        second = sensory(market)
        self.assertEqual(first.device.type, "cpu")
        self.assertTrue(torch.equal(first, second))

        # The fixed seed makes an independently constructed module identical too.
        twin = FrozenSensory(seed=SENSORY_SEED)
        self.assertTrue(torch.equal(sensory.projection, twin.projection))
        self.assertTrue(torch.equal(first, twin(market)))

        # The frozen representation does not depend on train/eval mode.
        self.assertTrue(torch.equal(first, sensory.eval()(market)))
        self.assertTrue(torch.equal(first, sensory.train()(market)))

    def test_02_shapes_are_explicit_and_malformed_shapes_fail(self):
        sensory = FrozenSensory()
        market = market_window()

        single = sensory(market)
        self.assertEqual(tuple(single.shape), (Z_DIM,))

        batch_market = market.unsqueeze(0).repeat(4, 1, 1)
        batch = sensory(batch_market)
        self.assertEqual(tuple(batch.shape), (4, Z_DIM))
        # Batched and single-sample GEMM paths may differ in the last float32
        # bit; the frozen mapping itself must agree to float tolerance.
        self.assertTrue(torch.allclose(batch[0], single, rtol=0.0, atol=1e-6))
        self.assertTrue(torch.equal(batch[0], batch[1]))

        # A different context length is legal only when the module was built for it.
        short = FrozenSensory(context_length=8)
        self.assertEqual(tuple(short(market_window(context_length=8)).shape), (Z_DIM,))

        malformed = {
            "four-dimensional": market.reshape(1, CONTEXT_LENGTH, MARKET_CHANNELS, 1),
            "one-dimensional": market.reshape(-1),
            "wrong channel count": market[:, :3],
            "wrong context length": market[: CONTEXT_LENGTH - 1],
            "wrong batched length": market.unsqueeze(0)[:, : CONTEXT_LENGTH - 1],
            "empty batch": market.unsqueeze(0)[:0],
            "integer dtype": torch.zeros(CONTEXT_LENGTH, MARKET_CHANNELS, dtype=torch.int64),
        }
        for label, bad in malformed.items():
            with self.subTest(label=label):
                with self.assertRaises(ContractError):
                    sensory(bad)
        with self.assertRaises(ContractError):
            sensory(market.numpy())
        with self.assertRaises(ContractError):
            sensory("not a tensor")

    def test_02_canonical_float64_normalized_tensor_is_accepted(self):
        # The canonical VS-A normalizer returns float64 NumPy; the frozen
        # projection owns the representation precision and casts at this boundary.
        normalized = normalize_market_window(canonical_raw_window())
        self.assertEqual(normalized.dtype, np.float64)

        sensory = FrozenSensory()
        single = sensory(torch.from_numpy(normalized))
        self.assertEqual(single.dtype, torch.float32)
        self.assertEqual(tuple(single.shape), (Z_DIM,))
        self.assertTrue(bool(torch.isfinite(single).all()))

        batch = sensory(torch.from_numpy(normalized).unsqueeze(0).repeat(3, 1, 1))
        self.assertEqual(tuple(batch.shape), (3, Z_DIM))
        self.assertTrue(torch.allclose(batch[0], single, rtol=0.0, atol=1e-6))

    def test_03_sensory_has_zero_trainable_parameters(self):
        sensory = FrozenSensory()
        self.assertEqual(list(sensory.parameters()), [])
        self.assertEqual(sum(parameter.numel() for parameter in sensory.parameters()), 0)
        self.assertFalse(sensory.projection.requires_grad)
        self.assertIn("projection", dict(sensory.named_buffers()))
        self.assertNotIn("projection", dict(sensory.named_parameters()))

    def test_05_non_finite_input_or_output_fails_explicitly(self):
        sensory = FrozenSensory()
        market = market_window()

        for bad_value in (float("nan"), float("inf"), float("-inf")):
            corrupted = market.clone()
            corrupted[0, 0] = bad_value
            with self.subTest(value=bad_value):
                with self.assertRaises(ContractError):
                    sensory(corrupted)

        # A finite float64 value beyond the projection dtype's range would cast
        # to infinity; it must fail instead of letting tanh saturate silently.
        overflow = torch.zeros(CONTEXT_LENGTH, MARKET_CHANNELS, dtype=torch.float64)
        overflow[0, 0] = 1e300
        with self.assertRaises(ContractError):
            sensory(overflow)

        # A non-finite projection (simulated buffer corruption) must not return
        # a silently NaN representation.
        sensory.projection[0, 0] = float("nan")
        with self.assertRaises(ContractError):
            sensory(market)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
