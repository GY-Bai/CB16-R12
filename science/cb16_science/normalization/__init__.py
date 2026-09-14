"""R12 normalization ablation (Science lane).

Public pieces:

* :mod:`cb16_science.normalization.data` -- read-only archive loading, raw
  validation and deterministic window selection;
* :mod:`cb16_science.normalization.transforms` -- pure N0..N4 transform
  functions;
* :mod:`cb16_science.normalization.diagnostics` -- invariant probes and the
  frozen sensory projection;
* :mod:`cb16_science.normalization.experiment` -- the allowlisted
  ``cb16.normalization-ablation@v1`` entrypoint and artifact writing.

The package is deliberately narrow: it must not learn, rank, promote, or reach
into dispatcher, GitHub, OCI, account or Central Brain code.
"""

from __future__ import annotations

from .contract import RESULT_COMMAND

__all__ = ["RESULT_COMMAND"]
