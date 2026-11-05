# DSS3 v4 BPE threshold sensitivity

## Methods

The fine-tuned Plant DNAGPT-BPE checkpoint was fixed. The positive-class prediction score was `softmax(logits)[1]`; scores above the respective threshold were called positive. Thresholds 0.3 through 0.9 (step 0.1) were specified before analysis. Dev and Test predictions were computed once and reused across all thresholds. No retraining occurred.

The decision threshold 0.5 reproduces the original reported Test metrics. The original user-adjustable web default was 0.8. Test-set analyses are descriptive, not used to optimize the threshold.

## Threshold-sensitivity results (held-out Test)

| Threshold | Accuracy | Precision | Recall | F1 | Predicted positives |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.7360 | 0.6731 | 0.9174 | 0.7765 | 2524 |
| 0.4 | 0.7686 | 0.7238 | 0.8688 | 0.7897 | 2223 |
| 0.5 | 0.7729 | 0.7612 | 0.7954 | 0.7779 | 1935 |
| 0.6 | 0.7716 | 0.8124 | 0.7063 | 0.7556 | 1610 |
| 0.7 | 0.7362 | 0.8531 | 0.5707 | 0.6839 | 1239 |
| 0.8 | 0.6774 | 0.8991 | 0.3996 | 0.5533 | 823 |
| 0.9 | 0.5845 | 0.9484 | 0.1787 | 0.3008 | 349 |

ROC-AUC: **0.8554**; PR-AUC (average precision): **0.8500**. Both are independent of the binary threshold.

## 0.5 versus 0.8

- At 0.5: precision 0.7612, recall 0.7954, F1 0.7779, predicted positives 1935.
- At 0.8: precision 0.8991, recall 0.3996, F1 0.5533, predicted positives 823.

## Interpretation

Scores are classifier outputs, NOT calibrated biological probabilities. The dataset distinguishes annotation-derived TSS-upstream candidates from gene-distal GC-matched weak negatives, not functional promoter strength or tissue specificity. The Test set must not be used to select a threshold.

Checkpoint: `<PROJECT_ROOT>/finetune/dss3_bpe_1000bp_head_only_v4_gc_seed42/best_model`

Data: `<PROJECT_ROOT>/DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched`
