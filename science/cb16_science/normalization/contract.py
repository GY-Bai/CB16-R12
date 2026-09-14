"""Frozen contract constants for the R12 normalization ablation.

Every value in this module is copied literally from
``docs/tasks/R12_NORMALIZATION_ABLATION_R0.md``.  The module is pure data: it
imports nothing, touches no market data, and knows nothing about GitHub, OCI,
the dispatcher, account state or Central Brain code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

SCHEMA_SPEC = "cb16.normalization_ablation.spec.v1"
SCHEMA_RESULT = "cb16.normalization_ablation.result.v1"
RESULT_COMMAND = "cb16.normalization-ablation@v1"
CONTRACT_FILE = "docs/tasks/R12_NORMALIZATION_ABLATION_R0.md"
LOGICAL_DATA_NAME = "binance_um_1m_klines_10pairs"
DEFAULT_KLINES_ROOT = "/cb16/raw/klines_1m"

# ---------------------------------------------------------------------------
# Fixed data scope
# ---------------------------------------------------------------------------

SYMBOL = "BTCUSDT"
FREQUENCY = "1m"
INTERVAL_MONTHS = ("2020-01", "2020-02", "2020-03")
INTERVAL_START_UTC = "2020-01-01T00:00:00Z"
INTERVAL_END_UTC = "2020-03-31T23:59:00Z"
CONTEXT_LENGTH = 64
MAX_EVALUATION_WINDOWS = 4096
PREDECESSOR_BARS = 1
WINDOW_SELECTION_METHOD = (
    "deterministic approximately-even spacing over all valid window endpoints; "
    "no random sampling; every candidate receives exactly the same endpoint set"
)

# ---------------------------------------------------------------------------
# Numeric contract
# ---------------------------------------------------------------------------

FLOAT_DTYPE = "float64"
EPS_V = 1e-12
EPS_STD = 1e-8
PRICE_SCALE_FACTOR = 100.0
VOLUME_SCALE_FACTOR = 1000.0
#: Absolute tolerance used only to turn a measured invariance difference into
#: the required boolean label.  The measured difference is always reported.
SCALE_INVARIANCE_ATOL = 1e-9

# ---------------------------------------------------------------------------
# Invariant probes
# ---------------------------------------------------------------------------

RUNTIME_WARMUPS = 1
RUNTIME_REPEATS = 3
CAUSALITY_PROBE_SAMPLE = 8
CAUSALITY_PRICE_MULTIPLIER = 1.37
CAUSALITY_PRICE_OFFSET = 0.5
CAUSALITY_VOLUME_MULTIPLIER = 2.5
CAUSALITY_VOLUME_OFFSET = 1.0
CANDLE_RANGE_REFERENCE = "raw log(H/L)"

# ---------------------------------------------------------------------------
# Frozen sensory diagnostic
# ---------------------------------------------------------------------------

PROJECTION_SEED = 24680
PROJECTION_BIT_GENERATOR = "PCG64"
PROJECTION_SHAPE = (32, 320)
PROJECTION_ROW_NORMALIZATION = "unit_l2"
PROJECTION_CHECKSUM_ALGORITHM = "sha256"

# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

CANDIDATE_IDS = ("N0", "N1", "N2", "N3", "N4")
OUTPUT_CHANNELS = ("pO", "pH", "pL", "pC", "v")

#: Declared, formula-derived interpretation of each candidate.  These are not
#: measurements and they are not a ranking: the measured invariant probes live
#: in ``RESULT.json`` under ``invariants``.  ``known_information_removed`` uses
#: only the enum values named in the task contract.
CANDIDATES = (
    {
        "candidate_id": "N0",
        "name": "endpoint-anchored log ratios",
        "role": "current R12 candidate",
        "description": (
            "pO/pH/pL/pC = log(channel_tau / C_t) for every represented bar and "
            "v = log(max(V_tau / median(V_window), eps_v))."
        ),
        "preserves_relative_volatility_amplitude": True,
        "invertible_to_relative_price_path_up_to_common_scale": True,
        "known_information_removed": ["window_location", "window_scale"],
    },
    {
        "candidate_id": "N1",
        "name": "per-bar log-return geometry",
        "role": "challenger",
        "description": (
            "pO/pH/pL/pC = log(channel_tau / C_{tau-1}) using the previous close, "
            "including the retained predecessor bar for the first represented bar; "
            "volume uses the same relative-window median formula as N0."
        ),
        "preserves_relative_volatility_amplitude": True,
        "invertible_to_relative_price_path_up_to_common_scale": True,
        "known_information_removed": ["window_location", "window_scale"],
    },
    {
        "candidate_id": "N2",
        "name": "causal window z-score with shared price statistics",
        "role": "challenger",
        "description": (
            "All four price channels are normalized with the SAME close-derived "
            "window statistics (x - mu_C) / sd_C; volume is z-scored on its own "
            "log scale.  sd_C < eps_std invalidates the price channels."
        ),
        "preserves_relative_volatility_amplitude": False,
        "invertible_to_relative_price_path_up_to_common_scale": True,
        "known_information_removed": ["window_location", "window_scale", "volatility_amplitude"],
    },
    {
        "candidate_id": "N3",
        "name": "RevIN-style per-channel instance normalization",
        "role": "deliberately literal representation-only RevIN-style challenger",
        "description": (
            "log(O), log(H), log(L), log(C) and log(max(V, eps_v)) are normalized "
            "independently per channel with that channel's own window mean and "
            "max(channel_std, eps_std); no learned affine and no denormalization."
        ),
        "preserves_relative_volatility_amplitude": False,
        "invertible_to_relative_price_path_up_to_common_scale": False,
        "known_information_removed": [
            "window_location",
            "window_scale",
            "volatility_amplitude",
            "per_channel_relative_geometry",
        ],
    },
    {
        "candidate_id": "N4",
        "name": "volatility-normalized return geometry",
        "role": "challenger",
        "description": (
            "The four N1 price-return channels are divided by sigma, the ddof=0 "
            "standard deviation of close-to-close log returns over the represented "
            "window (including the predecessor return); volume uses the same "
            "relative-window median formula as N0/N1.  sigma < eps_std invalidates "
            "the price channels."
        ),
        "preserves_relative_volatility_amplitude": False,
        "invertible_to_relative_price_path_up_to_common_scale": True,
        "known_information_removed": ["window_location", "window_scale", "volatility_amplitude"],
    },
)

# ---------------------------------------------------------------------------
# Interpretation label definitions
# ---------------------------------------------------------------------------

INTERPRETATION_LABELS = {
    "preserves_nominal_scale_invariance": (
        "Measured: the transformed output is unchanged within "
        f"{SCALE_INVARIANCE_ATOL:g} when every raw OHLC in the required source "
        f"window is multiplied by {PRICE_SCALE_FACTOR:g}."
    ),
    "preserves_ohlc_ordering": (
        "Measured: every represented bar still satisfies H >= max(O, C) and "
        "L <= min(O, C) in transformed coordinates; NaNs count as violations."
    ),
    "preserves_relative_volatility_amplitude": (
        "Declared from the fixed formula: the transform keeps raw per-window "
        "volatility magnitude information instead of rescaling each window to a "
        "fixed volatility unit."
    ),
    "invertible_to_relative_price_path_up_to_common_scale": (
        "Declared from the fixed formula: the transformed price channels determine "
        "the raw relative price path up to one positive common scale factor per "
        "window (not applicable when a channel-specific affine map is applied)."
    ),
    "known_information_removed": (
        "Declared from the fixed formula.  Enum values: window_location, "
        "window_scale, volatility_amplitude, per_channel_relative_geometry."
    ),
}

# ---------------------------------------------------------------------------
# Validation and reporting vocabulary
# ---------------------------------------------------------------------------

VALIDATION_RULES = (
    "reject non-finite OHLCV",
    "reject non-positive OHLC prices",
    "reject malformed OHLC ordering: H < max(O, C) or L > min(O, C)",
)

STATUS_OK = "OK"
STATUS_INPUT_UNAVAILABLE = "INPUT_UNAVAILABLE"
STATUS_EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"

CANDIDATE_STATUS_OK = "OK"
CANDIDATE_STATUS_PARTIAL = "PARTIAL"
CANDIDATE_STATUS_INVALID = "INVALID"

EXIT_OK = 0
EXIT_INPUT_UNAVAILABLE = 3
EXIT_EVIDENCE_INSUFFICIENT = 4

NO_WINNER_STATEMENT = "NO WINNER PROMOTED BY BUILDER"

SCOPE_NOTE = (
    "2020-01-01..2020-03-31 BTCUSDT 1m is exploratory normalization material "
    "only.  It is not a profitability result and not a holdout result."
)
