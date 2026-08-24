import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, classification_report


class OSRMetrics:
    """
        The seven open-set metrics reported for every method.

        `convention` selects two things at once, and is named for the method
        whose conventions are being followed rather than for the method being
        evaluated:

            "openmax"  unknown samples carry index 0, so every label is shifted
                       up by one; AUROC flips the outlier score to 1 - score.
            "opengan"  unknown samples carry index -1, and AUROC uses the
                       outlier score as given.
            anything   unknown samples carry index -1, and AUROC flips the
            else       score to 1 - score.

        Pass "opengan" whenever the score is already oriented so that a higher
        value means "more likely to be known".
    """

    def __init__(self, predict=None, label=None, outlier_scores=None, convention=None):
        self.outlier_scores = outlier_scores
        self.predict = predict
        self.label = label
        self.unknown_class_idx = None
        self.convention = convention

        if convention is None:
            raise ValueError("convention must be set: see the class docstring")
        elif convention == "openmax":
            # Shift the labels so that unknown moves from -1 to 0.
            self.unknown_class_idx = 0
            self.label = self.label + 1
        else:
            self.unknown_class_idx = -1

    def compute(self):
        return {
            "accuracy": self._accuracy(),
            "inner metric": self._inner_metric(),
            "UUC Accuracy": self._UUC_Accuracy(),
            "outer metric": self._outer_metric(),
            "halfpoint": self._halfpoint(),
            "F1 macro": self._f1_macro(),
            "auroc": self._AUROC(),
        }

    def _accuracy(self) -> float:
        """
        Returns the accuracy score of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        correct = (np.array(self.predict) == np.array(self.label)).sum()
        return float(correct) / float(len(self.predict))

    def _inner_metric(self) -> float:
        """Accuracy over KNOWN-class samples only (inner metric, or KKC accuracy)."""
        assert len(self.predict) == len(self.label)

        sample_indices = [i for i, (x, y) in enumerate(zip(self.predict, self.label))
                          if (y != self.unknown_class_idx and x != self.unknown_class_idx)]
        predictions = [self.predict[i] for i in sample_indices]

        correct = 0
        for prediction, idx in zip(predictions, sample_indices):
            if prediction == self.label[idx]:
                correct += 1

        if len(predictions) > 0:
            return float(correct) / float(len(predictions))

        return 1.0

    def _UUC_Accuracy(self) -> float:
        """
            Accuracy over UNKNOWN-class samples only (UUC accuracy).

            This is not the outer metric: it only asks how often an unknown
            sample was rejected, ignoring the known ones entirely.
        """
        assert len(self.predict) == len(self.label)

        sample_indices = [i for i, y in enumerate(self.label) if y == self.unknown_class_idx]
        predictions = [self.predict[i] for i in sample_indices]

        correct = 0
        for prediction, idx in zip(predictions, sample_indices):
            if prediction == self.unknown_class_idx:
                correct += 1

        if len(predictions) > 0:
            return float(correct) / float(len(predictions))

        return 1.0

    def _outer_metric(self) -> float:
        """
            How well the classifier separates knowns from unknowns, treated as
            a binary problem: which known class was predicted does not matter.
        """
        assert len(self.predict) == len(self.label)
        correct = 0

        for prediction, true_label in zip(self.predict, self.label):
            if true_label == self.unknown_class_idx:        # sample is UUC
                if prediction == self.unknown_class_idx:    # novelty was detected
                    correct += 1
            else:                                           # sample is KKC
                if prediction != self.unknown_class_idx:    # accepted as known, right class or not
                    correct += 1

        return float(correct) / float(len(self.predict))

    def _halfpoint(self) -> float:
        """
            A variant of the inner metric that also counts false unknowns: a
            known sample wrongly rejected is scored as an error rather than
            dropped from the denominator.
        """
        assert len(self.predict) == len(self.label)

        sample_indices = [i for i, y in enumerate(self.label) if y != self.unknown_class_idx]
        predictions = [self.predict[i] for i in sample_indices]

        correct = 0
        for prediction, idx in zip(predictions, sample_indices):
            if prediction == self.label[idx]:
                correct += 1

        return float(correct) / float(len(predictions))

    def _f1_macro(self) -> float:
        """
        Returns the F1-measure with a macro average of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='macro')

    def _AUROC(self) -> float:
        """Area under the ROC curve for the known/unknown decision."""
        if self.outlier_scores is None:
            return
        assert len(self.outlier_scores) == len(self.label)
        label = np.array(self.label)
        y_true_bin = (label != self.unknown_class_idx).astype(int)

        if self.convention == "opengan":
            y_score = self.outlier_scores
        else:
            y_score = 1 - self.outlier_scores

        return roc_auc_score(y_true_bin, y_score)

    def per_class_metrics(self):
        return classification_report(y_true=self.label, y_pred=self.predict,
                                     output_dict=True, zero_division=0)
