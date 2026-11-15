# ProMoAI experiment scripts

These scripts implement project-specific *Salvia* experiments using pretrained Plant DNA LLMs. They are not a copy of the complete upstream repository. See `../upstream.md` for model provenance and licensing.

## Suggested order

1. `build_dss3_chromosome_holdout_v4_gc_matched.py`: construct a GC-matched chromosome-holdout dataset from FASTA/GFF files that you are authorized to use.
2. `train_dss3_bpe_1000bp_head_only.py`: freeze the BPE backbone and train only the classification head.
3. `evaluate_dss3_bpe_1000bp.py`: evaluate BPE on the fixed test chromosomes.
4. `train_dss3_6mer_1000bp_head_only_v4_gc.py`: freeze the 6-mer backbone and train only the classification head.
5. `evaluate_dss3_6mer_1000bp_v4_gc.py`: evaluate the 6-mer model.
6. `analyze_dss3_v4_threshold_sensitivity.py`: analyze threshold sensitivity using fixed prediction scores.
