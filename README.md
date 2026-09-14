# CB16-R12

Clean reconstruction line for Central Brain / CB16.

This repository starts with a minimal AI-assisted development system. R11 is reference material, not an automatic implementation base.

Initial roles:
- Master: product/scientific authority and trade-off decisions.
- Chat-SOL: architecture, mathematical/semantic specification, task decomposition, PR review, acceptance/rejection.
- GitHub: control plane and durable communication boundary.
- OCI: canonical execution environment.
- DeepSeek API Builder: bounded implementation and testing.

Core rule: GitHub carries durable intent and observable execution facts; OCI performs execution; Chat-SOL does not require direct shell access to OCI.
