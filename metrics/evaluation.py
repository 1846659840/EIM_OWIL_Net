from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, normalized_mutual_info_score, roc_auc_score


def _flat(x):
    return np.asarray(x).reshape(-1)


def _safe_auc(y_true, y_score):
    y_true = _flat(y_true)
    y_score = _flat(y_score)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


class VADMetrics:
    """Frame-level detection metrics from the appendix."""

    @staticmethod
    def compute_roc_auc(y_true, y_pred):
        return _safe_auc(y_true, y_pred)

    @staticmethod
    def compute_ap(y_true, y_pred):
        y_true = _flat(y_true)
        y_pred = _flat(y_pred)
        if y_true.size == 0 or np.unique(y_true).size < 2:
            return 0.0
        return float(average_precision_score(y_true, y_pred))

    @staticmethod
    def compute_far_at_0_5(y_true, y_pred):
        y_true = _flat(y_true).astype(int)
        y_pred = _flat(y_pred)
        normal = y_true == 0
        if normal.sum() == 0:
            return 0.0
        return float(((y_pred > 0.5) & normal).sum() / normal.sum())

    @staticmethod
    def compute_all(y_true, y_pred):
        return {
            "roc_auc": VADMetrics.compute_roc_auc(y_true, y_pred),
            "ap": VADMetrics.compute_ap(y_true, y_pred),
            "far_at_0_5": VADMetrics.compute_far_at_0_5(y_true, y_pred),
        }


class OpenWorldMetrics:
    """Known-AUC, Unknown-AUROC, OSCR, H-score, and NMI."""

    @staticmethod
    def compute_oscr(y_true, pred_class, known_confidence, known_classes):
        y_true = _flat(y_true)
        pred_class = _flat(pred_class)
        known_confidence = _flat(known_confidence)
        known_mask = np.isin(y_true, known_classes)
        unknown_mask = ~known_mask
        if known_mask.sum() == 0 or unknown_mask.sum() == 0:
            return 0.0

        correct_known = (pred_class[known_mask] == y_true[known_mask]).astype(float)
        known_scores = known_confidence[known_mask]
        unknown_scores = known_confidence[unknown_mask]
        thresholds = np.r_[np.inf, np.sort(np.unique(known_confidence))[::-1], -np.inf]
        ccr = []
        fpr = []
        for threshold in thresholds:
            ccr.append(float(((known_scores >= threshold) * correct_known).sum() / known_mask.sum()))
            fpr.append(float((unknown_scores >= threshold).sum() / unknown_mask.sum()))
        order = np.argsort(fpr)
        return float(np.trapz(np.asarray(ccr)[order], np.asarray(fpr)[order]))

    @staticmethod
    def compute_h_score(known_acc, unknown_acc):
        denom = known_acc + unknown_acc
        if denom <= 0:
            return 0.0
        return float(2 * known_acc * unknown_acc / denom)

    @staticmethod
    def compute_all(
        y_true_class,
        pred_class,
        known_confidence,
        known_classes,
        unknown_score=None,
        pred_is_unknown=None,
        cluster_labels=None,
        cluster_true_labels=None,
        known_binary_labels=None,
        known_scores=None,
        known_auc_mask=None,
    ):
        y_true_class = _flat(y_true_class)
        pred_class = _flat(pred_class)
        known_confidence = _flat(known_confidence)
        known_mask = np.isin(y_true_class, known_classes)
        unknown_mask = ~known_mask
        if unknown_score is None:
            unknown_score = 1.0 - known_confidence
        unknown_score = _flat(unknown_score)
        if pred_is_unknown is None:
            pred_is_unknown = unknown_score >= 0.5
        pred_is_unknown = _flat(pred_is_unknown).astype(bool)

        if known_binary_labels is not None and known_scores is not None:
            if known_auc_mask is not None:
                known_auc_mask = _flat(known_auc_mask).astype(bool)
                known_binary_labels = _flat(known_binary_labels)[known_auc_mask]
                known_scores = _flat(known_scores)[known_auc_mask]
            known_auc = _safe_auc(known_binary_labels, known_scores)
        else:
            known_auc = _safe_auc(known_mask.astype(int), known_confidence)
        unknown_auroc = _safe_auc(unknown_mask.astype(int), unknown_score)
        oscr = OpenWorldMetrics.compute_oscr(
            y_true_class, pred_class, known_confidence, known_classes
        )

        if known_mask.sum() > 0:
            known_acc = float(
                ((pred_class[known_mask] == y_true_class[known_mask]) & ~pred_is_unknown[known_mask]).mean()
            )
        else:
            known_acc = 0.0
        unknown_acc = float(pred_is_unknown[unknown_mask].mean()) if unknown_mask.sum() > 0 else 0.0
        h_score = OpenWorldMetrics.compute_h_score(known_acc, unknown_acc)

        if cluster_labels is not None:
            cluster_labels = _flat(cluster_labels)
            y_cluster = _flat(cluster_true_labels) if cluster_true_labels is not None else y_true_class[unknown_mask]
            n = min(cluster_labels.size, y_cluster.size)
            if n > 1:
                nmi = float(normalized_mutual_info_score(y_cluster[:n], cluster_labels[:n]))
            else:
                nmi = 0.0
        else:
            nmi = 0.0

        return {
            "known_auc": known_auc,
            "unknown_auroc": unknown_auroc,
            "oscr": oscr,
            "h_score": h_score,
            "nmi": nmi,
        }


class IncrementalMetrics:
    """Metrics over A_{i,j}, the AUC on task j after training task i."""

    def __init__(self, num_tasks, memory_bytes=0):
        self.num_tasks = num_tasks
        self.memory_bytes = memory_bytes
        self.task_matrix = defaultdict(dict)
        self.random_baseline = {}

    def update(self, task_trained, task_eval, metric_value):
        self.task_matrix[int(task_trained)][int(task_eval)] = float(metric_value)

    def update_random_baseline(self, task_eval, metric_value):
        self.random_baseline[int(task_eval)] = float(metric_value)

    def compute_average_performance(self):
        if not self.task_matrix:
            return 0.0
        final_task = max(self.task_matrix.keys())
        values = [self.task_matrix[final_task][j] for j in range(final_task + 1)
                  if j in self.task_matrix[final_task]]
        return float(np.mean(values)) if values else 0.0

    def compute_bwt(self):
        if not self.task_matrix:
            return 0.0
        final_task = max(self.task_matrix.keys())
        values = []
        for j in range(final_task):
            if j in self.task_matrix[final_task] and j in self.task_matrix.get(j, {}):
                values.append(self.task_matrix[final_task][j] - self.task_matrix[j][j])
        return float(np.mean(values)) if values else 0.0

    def compute_fwt(self):
        values = []
        for j in range(1, self.num_tasks):
            if j in self.task_matrix.get(j - 1, {}):
                baseline = self.random_baseline.get(j, 0.0)
                values.append(self.task_matrix[j - 1][j] - baseline)
        return float(np.mean(values)) if values else 0.0

    def compute_forgetting(self):
        if not self.task_matrix:
            return 0.0
        final_task = max(self.task_matrix.keys())
        values = []
        for j in range(final_task):
            if j not in self.task_matrix[final_task]:
                continue
            best = max(self.task_matrix[i][j] for i in self.task_matrix if j in self.task_matrix[i])
            values.append(best - self.task_matrix[final_task][j])
        return float(np.mean(values)) if values else 0.0

    def compute_all(self):
        avg_auc = self.compute_average_performance()
        forget = self.compute_forgetting()
        return {
            "avg_auc": avg_auc,
            "bwt": self.compute_bwt(),
            "fwt": self.compute_fwt(),
            "forget": forget,
            "mem_mb": self.memory_bytes / (1024 ** 2),
        }


class ExplanationMetrics:
    """Faithfulness metrics from top-K interaction perturbation curves."""

    @staticmethod
    def relative_drop(original_scores, modified_scores, eps=1e-8):
        original = np.asarray(original_scores, dtype=float)
        modified = np.asarray(modified_scores, dtype=float)
        return float(np.mean((original - modified) / (np.abs(original) + eps)))

    @staticmethod
    def compute_drop_at_k(original_scores, modified_scores, k=5):
        return ExplanationMetrics.relative_drop(original_scores, modified_scores)

    @staticmethod
    def compute_aopc(original_scores, deletion_scores):
        original = np.asarray(original_scores, dtype=float)
        drops = [np.mean(original - np.asarray(scores, dtype=float)) for scores in deletion_scores]
        return float(np.mean(drops))

    @staticmethod
    def compute_sufficiency(original_scores, only_top_scores, eps=1e-8):
        original = np.asarray(original_scores, dtype=float)
        only_top = np.asarray(only_top_scores, dtype=float)
        return float(np.mean(only_top / (np.abs(original) + eps)))

    @staticmethod
    def compute_comprehensiveness(original_scores, deletion_by_k, ks=(1, 3, 5)):
        values = []
        for k in ks:
            if k < len(deletion_by_k):
                values.append(ExplanationMetrics.relative_drop(original_scores, deletion_by_k[k]))
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def compute_insdel(insertion_scores, deletion_scores):
        ins = np.asarray([np.mean(s) for s in insertion_scores], dtype=float)
        delete = np.asarray([np.mean(s) for s in deletion_scores], dtype=float)
        x = np.linspace(0.0, 1.0, len(ins))
        return float(np.trapz(ins, x) - np.trapz(delete, x))

    @staticmethod
    def compute_from_curves(curves):
        full = curves["full"]
        deletion = curves["deletion"]
        insertion = curves["insertion"]
        return {
            "drop_at_k": ExplanationMetrics.compute_drop_at_k(full, curves["without_top"]),
            "aopc": ExplanationMetrics.compute_aopc(full, deletion),
            "suff": ExplanationMetrics.compute_sufficiency(full, curves["only_top"]),
            "comp": ExplanationMetrics.compute_comprehensiveness(full, deletion),
            "insdel": ExplanationMetrics.compute_insdel(insertion, deletion),
        }


def evaluate_full(outputs_list, labels_list, classes_list, cfg):
    scores = []
    labels = []
    classes = []
    pred_classes = []
    confidences = []
    unknown_scores = []
    pred_unknown = []

    for output, frame_labels, anomaly_classes in zip(outputs_list, labels_list, classes_list):
        s_t = output["s_t"].detach().cpu().numpy()
        y_t = frame_labels.detach().cpu().numpy()
        cls = anomaly_classes.detach().cpu().numpy()
        scores.append(s_t.reshape(-1))
        labels.append((y_t.reshape(-1) > 0.5).astype(int))
        classes.append(np.repeat(cls.reshape(-1), s_t.shape[1]))

        if "open_world" in output:
            ow = output["open_world"]
            pred_classes.append(ow["pred_class"].detach().cpu().numpy().reshape(-1))
            confidences.append(ow["max_confidence"].detach().cpu().numpy().reshape(-1))
            if "unknown_score" in ow:
                unknown_scores.append(ow["unknown_score"].detach().cpu().numpy().reshape(-1))
            else:
                residual = ow["residual"].detach().cpu().numpy().reshape(-1)
                energy = ow["energy"].detach().cpu().numpy().reshape(-1)
                conf = ow["max_confidence"].detach().cpu().numpy().reshape(-1)
                unknown_scores.append(residual + energy - conf)
            pred_unknown.append(ow["is_unknown"].detach().cpu().numpy().reshape(-1))

    all_scores = np.concatenate(scores) if scores else np.array([])
    all_labels = np.concatenate(labels) if labels else np.array([])
    vad_metrics = VADMetrics.compute_all(all_labels, all_scores)

    open_metrics = {}
    if pred_classes:
        all_classes = np.concatenate(classes)
        known_auc_mask = (all_labels == 0) | (
            (all_labels == 1) & np.isin(all_classes, cfg.data.seen_classes)
        )
        open_metrics = OpenWorldMetrics.compute_all(
            all_classes,
            np.concatenate(pred_classes),
            np.concatenate(confidences),
            cfg.data.seen_classes,
            unknown_score=np.concatenate(unknown_scores),
            pred_is_unknown=np.concatenate(pred_unknown),
            known_binary_labels=all_labels,
            known_scores=all_scores,
            known_auc_mask=known_auc_mask,
        )

    return {"vad": vad_metrics, "open_world": open_metrics}
