"""CB16-R12 science-lane entrypoints.

Only identifiers listed in ``config/cb16_science_allowlist.json`` may be
executed by the Science lane.  Entrypoints must be deterministic: the same
commit plus the same inputs must produce byte-identical artifacts.
"""
