import os

import numpy as np
import pandas as pd

from Utils import OSRMetrics, CumulativeOSRConfusionMatrix as mc


class MetricLogger:
    """
        Collects and aggregates open-set metrics across folds and thresholds.

        Metrics (accuracy, F1, AUROC, ...) are accumulated per epsilon and per
        fold, alongside confusion matrices accumulated over the folds. On
        aggregate() it writes:
          - one consolidated CSV with mean and standard deviation per epsilon;
          - one CSV per fold, under a "Folds" subdirectory;
          - the accumulated confusion matrices per epsilon (when enabled).

        Args:
            epsilons: thresholds to evaluate.
            n_folds: number of cross-validation folds.
            dir: directory the CSVs and matrices are written to.
            flag_mc: whether to keep and export accumulated confusion matrices.
            mc_column_names: column labels for the confusion matrix.
            mc_title: title drawn on the confusion matrix.
            predict_unknown_value: the index that marks an unknown prediction.
            epsilon_decimals: how many decimals the epsilon keys are rounded to.
                Rounding exists only to normalize float noise from np.arange;
                it must stay finer than the sweep's step, or distinct epsilons
                collapse into one key and get averaged together. The default of
                6 is finer than any sweep used here.
    """

    def __init__(self, epsilons, n_folds, dir, flag_mc=True, mc_column_names=None,
                 mc_title=None, predict_unknown_value=-1, epsilon_decimals=6):
        self.epsilon_decimals = epsilon_decimals
        self.epsilons = [round(e, epsilon_decimals) for e in epsilons]
        self.flag_mc = flag_mc
        self.dir = dir
        self.n_folds = n_folds
        self.mc_column_names = mc_column_names
        self.mc_title = mc_title
        self.predict_unknown_value = predict_unknown_value

        self.METRIC_KEYS = ("F1 macro", "accuracy", "UUC Accuracy", "inner metric",
                            "outer metric", "halfpoint", "auroc")

        self.results_by_epsilon = {e: {key: [] for key in self.METRIC_KEYS} for e in self.epsilons}
        if n_folds > 0:
            self.results_by_fold = {fold: [] for fold in range(n_folds)}
        else:
            self.results_by_fold = {0: []}

        if flag_mc:
            self.accumulated_confusion_matrices = {e: None for e in self.epsilons}

    def update(self, metrics, fold, epsilon):
        """
            Records the metrics of one run, meaning one (fold, epsilon) pair.

            Args:
                metrics: the computed metrics, keyed by METRIC_KEYS.
                fold: index of the current fold.
                epsilon: threshold used in this run.
        """
        epsilon = round(epsilon, self.epsilon_decimals)
        current_epsilon_fold_data = {"epsilon": epsilon}

        for metric in self.METRIC_KEYS:
            self.results_by_epsilon[epsilon][metric].append(metrics[metric])
            current_epsilon_fold_data[metric] = metrics[metric]

        self.results_by_fold[fold].append(current_epsilon_fold_data)

    def update_mc(self, epsilon, predicts, targets, original_targets):
        """
            Folds one run's predictions into the confusion matrix of this
            epsilon, creating the matrix on first use.

            Args:
                epsilon: threshold these predictions belong to.
                predicts: the classifier's predictions.
                targets: target labels, with -1 marking unknown.
                original_targets: labels before remapping.
        """
        epsilon = round(epsilon, self.epsilon_decimals)

        if self.accumulated_confusion_matrices[epsilon] is None:
            matrix = mc(predicts, targets, original_targets, [], self.mc_column_names,
                        self.mc_title, predict_unknown_value=self.predict_unknown_value)
            matrix.compute()
            self.accumulated_confusion_matrices[epsilon] = matrix
        else:
            self.accumulated_confusion_matrices[epsilon].set_data(predicts, targets, original_targets)
            self.accumulated_confusion_matrices[epsilon].compute()

    def aggregate(self, csv_name):
        """
            Consolidates every recorded run and writes the output files: mean
            and standard deviation of each metric per epsilon, one CSV per
            fold, and the accumulated confusion matrices.

            Args:
                csv_name: filename of the consolidated CSV.

            Returns:
                The consolidated frame at full precision -- one row per
                epsilon, with the mean and standard deviation of every metric.
        """
        final_data = []
        os.makedirs(self.dir, exist_ok=True)

        for epsilon in sorted(self.results_by_epsilon.keys()):
            metrics = self.results_by_epsilon[epsilon]
            row = {"epsilon": epsilon}

            for metric in self.METRIC_KEYS:
                row[f"{metric}_mean"] = np.mean(metrics[metric])
                row[f"{metric}_std"] = np.std(metrics[metric])

            final_data.append(row)

            if self.flag_mc:
                self.accumulated_confusion_matrices[epsilon].save(
                    dir=os.path.join(self.dir, "confusion_matrices"), name=f"epsilon_{epsilon}")

        for fold in range(self.n_folds):
            df_fold = pd.DataFrame(self.results_by_fold[fold])
            fold_dir = os.path.join(self.dir, "Folds")
            os.makedirs(fold_dir, exist_ok=True)
            fold_filename = f"Results_Fold_{fold}.csv"
            fold_metric_cols = [c for c in df_fold.columns if c != "epsilon"]
            df_fold[fold_metric_cols] = df_fold[fold_metric_cols].round(3)
            df_fold.to_csv(os.path.join(fold_dir, fold_filename), index=False)
            print(f"[*] wrote fold {fold}: {fold_filename}")

        df = pd.DataFrame(final_data)
        csv_path = os.path.join(self.dir, csv_name)
        # The metric columns are written at 3 decimals, but epsilon keeps its
        # full precision: rounding it here would merge rows of a fine sweep.
        metric_cols = [c for c in df.columns if c != "epsilon"]
        df.round({c: 3 for c in metric_cols}).to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")

        # Returned unrounded: three decimals tie hundreds of rows of a fine
        # sweep, so a caller picking a best row needs the full precision.
        return df
