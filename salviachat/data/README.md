# Runtime data

This directory contains `SalviaBench-Final-150.csv`, a frozen 150-question benchmark export with `Category`, `Question`, and `Answer` columns. It is evaluation data, not a runtime dependency of the SalviaChat API. Do not tune the system using its gold answers after the first formal evaluation.

The source benchmark comprises 100 test cases and 50 challenge cases, including 30 newly constructed replacements. This three-column CSV does not retain split labels or case IDs; use the original benchmark manifest for split-specific analyses.

Other runtime data and databases are not committed to Git. See [model assets](../../docs/model_assets.md) for the benchmark checksum and external resource paths.
