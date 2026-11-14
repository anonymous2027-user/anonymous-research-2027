"""Positive F1 thresholding, tie-aware ROC-AUC, and non-interpolated AP."""
from fractions import Fraction
import numpy as np


def checked(y, score):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or not len(y):
        raise ValueError('Expected nonempty, aligned one-dimensional labels and scores')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all() or len(np.unique(y)) != 2:
        raise ValueError('Need finite scores and both binary classes')
    return y.astype(int), score


def choose_threshold(y, score, split='dev'):
    if split != 'dev':
        raise ValueError('Threshold selection is allowed only on Dev')
    y, score = checked(y, score)
    candidates = np.unique(np.r_[0.0, np.nextafter(score.min(), -np.inf), score])

    def key(threshold):
        prediction = score > threshold
        tp = int(np.sum(prediction & (y == 1)))
        fp = int(np.sum(prediction & (y == 0)))
        fn = int(np.sum(~prediction & (y == 1)))
        return Fraction(2*tp, max(2*tp+fp+fn, 1)), -fp, -abs(float(threshold)), float(threshold)
    return float(max(candidates, key=key))


def metrics(y, score, threshold):
    y, score = checked(y, score)
    if not np.isfinite(threshold):
        raise ValueError('Nonfinite threshold')
    prediction = score > threshold
    tp = int(np.sum(prediction & (y == 1)))
    fp = int(np.sum(prediction & (y == 0)))
    tn = int(np.sum(~prediction & (y == 0)))
    fn = int(np.sum(~prediction & (y == 1)))
    f1 = 2*tp/max(2*tp+fp+fn, 1)
    negative_f1 = 2*tn/max(2*tn+fp+fn, 1)
    positive, negative = score[y == 1], score[y == 0]
    comparisons = positive[:, None]-negative[None, :]
    auc = float(((comparisons > 0)+0.5*(comparisons == 0)).mean())
    order = np.argsort(-score, kind='stable')
    ordered_y, ordered_score = y[order], score[order]
    boundaries = np.r_[np.flatnonzero(np.diff(ordered_score)), len(y)-1]
    true_positives = np.cumsum(ordered_y)[boundaries]
    precision = true_positives/(boundaries+1)
    recall = true_positives/y.sum()
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall])*precision))
    return dict(threshold=float(threshold), f1_positive=f1, f1_macro=(f1+negative_f1)/2,
        roc_auc=auc, average_precision=average_precision, precision=tp/max(tp+fp, 1),
        recall=tp/max(tp+fn, 1), tn=tn, fp=fp, fn=fn, tp=tp)


def summarize_scores(scores, manifest, split, threshold=None):
    required = {'sample_id', 'participant_id', 'true_label', 'nll_margin'}
    if not required <= set(scores) or scores.sample_id.duplicated().any():
        raise ValueError('Invalid or duplicate window scores')
    if set(scores.sample_id) != set(manifest.sample_id):
        raise ValueError('Evaluation scores do not cover exactly the manifest')
    aligned = manifest.set_index('sample_id').loc[scores.sample_id]
    if not np.array_equal(aligned.participant_id.astype(str).to_numpy(), scores.participant_id.astype(str)):
        raise ValueError('Score participant mismatch')
    if not np.array_equal(aligned.label.to_numpy(), scores.true_label.to_numpy()):
        raise ValueError('Score label mismatch')
    if not np.isfinite(scores.nll_margin).all():
        raise ValueError('Nonfinite window score')
    people = scores.groupby('participant_id', as_index=False).agg(
        label=('true_label', 'first'), windows=('sample_id', 'size'), score=('nll_margin', 'mean'))
    if not people.windows.eq(200).all():
        raise ValueError('Expected exactly 200 scores per participant')
    if threshold is None:
        threshold = choose_threshold(people.label, people.score, split)
    result = metrics(people.label, people.score, threshold)
    result['split'] = split
    return people, result
