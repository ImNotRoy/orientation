import numpy as np


def class_precision(pred: np.ndarray, gt: np.ndarray):
    keys = sorted(np.unique(gt))
    r = [0] * len(np.unique(gt))
    for idx, cls in enumerate(keys):
        mask = gt == cls
        correct = pred == gt
        correct = np.bitwise_and(correct, mask)
        r[idx] = np.sum(correct) / np.sum(mask)
    return np.array(r)


def precision(pred: np.ndarray, gt: np.ndarray):
    cls_prec = class_precision(pred, gt)
    cls_prec = np.where(np.isnan(cls_prec), 0.0, cls_prec)
    return np.average(cls_prec)


def class_recall(pred: np.ndarray, gt: np.ndarray):
    keys = sorted(np.unique(gt))
    r = [0] * len(np.unique(gt))
    for idx, cls in enumerate(keys):
        mask = pred == cls
        correct = pred == gt
        correct = np.bitwise_and(correct, mask)
        r[idx] = np.sum(correct) / np.sum(mask)
    return np.array(r)


def recall(pred: np.ndarray, gt: np.ndarray):
    cls_rec = class_recall(pred, gt)
    cls_rec = np.where(np.isnan(cls_rec), 0.0, cls_rec)
    return np.average(cls_rec)


def class_f1(pred: np.ndarray, gt: np.ndarray):
    cls_prec = class_precision(pred, gt)
    cls_rec = class_recall(pred, gt)
    eps = 1e-8
    return 2 * cls_prec * cls_rec / (cls_prec + cls_rec + eps)


def f1(pred: np.ndarray, gt: np.ndarray):
    cls_f1 = class_f1(pred, gt)
    cls_f1 = np.where(np.isnan(cls_f1), 0.0, cls_f1)
    return np.average(cls_f1)


def get_metrics(pred: np.ndarray, gt: np.ndarray):
    assert len(pred) == len(gt)
    return {
        "precision": precision(pred, gt),
        "class_prec": class_precision(pred, gt),
        "recall": recall(pred, gt),
        "class_rec": class_recall(pred, gt),
        "f1": f1(pred, gt),
        "class_f1": class_f1(pred, gt),
    }
