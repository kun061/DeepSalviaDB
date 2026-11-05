# Upstream relationship

ProMoAI is not a new foundation model trained from scratch. It adapts the pretrained Plant DNAGPT-BPE and Plant DNAGPT-6mer checkpoints to a DSS3 1-kb promoter-candidate classification task.

This repository contains the project-specific parts:

- DSS3 promoter-candidate and GC-matched background construction;
- chromosome-held-out split;
- classification-head-only fine-tuning;
- independent evaluation and threshold analysis;
- REST inference integration.

It intentionally does not copy the complete Plant DNA LLMs source repository or pretrained checkpoints. Download the checkpoints from their official Hugging Face pages and cite the PDLLMs paper. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

For reproducibility, record the exact upstream Git commit and Hugging Face model revision used for each released ProMoAI version.
