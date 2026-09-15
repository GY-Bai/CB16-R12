# CB16-R12 Documentation Authority Map

This directory is intentionally partitioned by authority. **Do not ingest `docs/` recursively as if every file were an equally current rule.**

## Default reading order

1. `authority/` — current canonical governance and scientific interpretation.
2. `design/current/` — current design candidates; subordinate to canonical authority.
3. `operations/current/` — current execution/infra contracts; not product-science authority.
4. `tasks/active/` — the bounded task currently authorized for execution, when one exists.
5. `experiments/frozen/` — historical frozen experiment/task contracts; read only for the named historical run or dependency.
6. `archive/` — superseded/archaeological material; never default execution context.
7. `templates/` — examples only; never authority.

## Temporal rule

Current canonical authority governs present interpretation. Frozen historical contracts preserve what was actually specified at that time. Later interpretation may narrow propagation of an old result but must not rewrite its original gate or observation.

If two documents conflict, do not count files or choose the more detailed wording. Resolve by authority class, explicit dependency, and chronology.

Start with `authority/README.md`.