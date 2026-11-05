"""Binary metrics and vertex-weight bootstrap; NumPy only, no TensorFlow."""
import numpy as np


def binary_metrics(y, probability, threshold=0.5, weights=None):
    y = np.asarray(y, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    if y.ndim != 1 or probability.shape != y.shape or not len(y):
        raise ValueError('Expected nonempty one-dimensional labels/probabilities')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probability).all():
        raise ValueError('Invalid labels or nonfinite probabilities')
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError('Probabilities must be in [0,1]')
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('Invalid sample weights')
    positive, negative = float(np.dot(w, y)), float(np.dot(w, 1 - y))
    if positive <= 0 or negative <= 0:
        raise ValueError('Both classes must have positive weight')
    order = np.argsort(-probability, kind='stable')
    sy, sp, sw = y[order], probability[order], w[order]
    ends = np.r_[np.flatnonzero(np.diff(sp)), len(y) - 1]
    tp_curve = np.cumsum(sw * sy)[ends]
    fp_curve = np.cumsum(sw * (1 - sy))[ends]
    recall = np.r_[0.0, tp_curve / positive]
    fpr = np.r_[0.0, fp_curve / negative]
    precision_tail = np.divide(tp_curve, tp_curve + fp_curve,
                               out=np.ones_like(tp_curve), where=(tp_curve + fp_curve) > 0)
    precision_curve = np.r_[1.0, precision_tail]
    prediction = probability >= threshold
    tp = float(w[(y == 1) & prediction].sum())
    fp = float(w[(y == 0) & prediction].sum())
    tn = float(w[(y == 0) & ~prediction].sum())
    fn = float(w[(y == 1) & ~prediction].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / positive
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics = {
        'roc_auc': float(np.trapz(recall, fpr)),
        'average_precision': float(np.dot(np.diff(recall), precision_tail)),
        'pr_auc_trapezoid': float(np.trapz(precision_curve, recall)),
        'precision': precision, 'recall': sensitivity,
        'f1': 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0,
        'accuracy': (tp + tn) / (positive + negative),
        'specificity': tn / negative,
        'mcc': (tp * tn - fp * fn) / denom if denom else 0.0,
        'positive_prevalence': positive / (positive + negative),
        'threshold': float(threshold), 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }
    curves = {'fpr': fpr, 'tpr': recall, 'recall': recall, 'precision': precision_curve}
    return metrics, curves


def vertex_bootstrap(y, probability, group_a, group_b, repeats=1000, seed=20261018):
    """Approximate vertex-bootstrap sensitivity intervals at fixed threshold 0.5.

    Sample N protein-input groups with replacement, then weight edge (a,b) by
    count(a)*count(b). Shared proteins therefore have shared resampling weights.
    Conditional on this fitted model and split; not a universal network CI or
    uncertainty over retraining, homologs, label noise or species sampling.
    """
    if repeats < 1:
        raise ValueError('Bootstrap repeats must be positive')
    groups = sorted(set(group_a) | set(group_b))
    index = {group: i for i, group in enumerate(groups)}
    ia, ib = np.array([index[g] for g in group_a]), np.array([index[g] for g in group_b])
    if len(ia) != len(y) or len(ib) != len(y):
        raise ValueError('Bootstrap endpoint count mismatch')
    rng = np.random.default_rng(seed)
    keys = ('roc_auc', 'average_precision', 'precision', 'recall', 'f1')
    draws, skipped = [], 0
    for replicate in range(repeats):
        multiplicity = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        weights = multiplicity[ia] * multiplicity[ib]
        try:
            metrics, _ = binary_metrics(y, probability, weights=weights)
        except ValueError as exc:
            if str(exc) != 'Both classes must have positive weight':
                raise
            skipped += 1
            continue
        draws.append({'replicate': replicate, **{k: metrics[k] for k in keys}})
    if len(draws) < max(10, int(repeats * .8)):
        raise ValueError('Too few valid vertex-bootstrap replicates')
    intervals = {k: {'lower_2_5': float(np.percentile([r[k] for r in draws], 2.5)),
                     'upper_97_5': float(np.percentile([r[k] for r in draws], 97.5))} for k in keys}
    return {'method': 'vertex_weight_bootstrap_percentile', 'threshold': 0.5,
            'requested_replicates': repeats, 'valid_replicates': len(draws),
            'skipped_single_class': skipped, 'seed': seed, 'protein_input_groups': len(groups),
            'intervals': intervals,
            'scope': 'Approximate sensitivity intervals conditional on fixed split/model; not retraining or label uncertainty.'}, draws
