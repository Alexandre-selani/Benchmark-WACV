import os.path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


class CumulativeOSRConfusionMatrix:
    """
        Confusion matrix accumulated across folds, with a dedicated row for
        rejected (unknown) predictions.

        Columns are the true classes and rows are the predictions, row 0 being
        "unknown".
    """

    def __init__(self, predict, target_test, target_original, UUC_classes, col_labels,
                 title=None, predict_unknown_value=-1):
        # target_test/target_original are always raw labels (unknown = -1, known
        # classes = 0..N-1), so they are shifted by +1 into the index space used
        # by the rows and columns of the matrix.
        #
        # `predict` varies by convention: most methods use the same raw
        # convention (-1 = unknown), while OpenMax already returns the index of
        # its own score column (0 = unknown, 1..N = classes), which IS that
        # space. predict_unknown_value says which value the convention uses for
        # "unknown", and subtracting it aligns either convention with
        # target_test/target_original.
        self.predict_unknown_value = predict_unknown_value
        self.predict = predict - predict_unknown_value
        # +1 because targets can be -1 when an entire dataset is used as the
        # unknown set alongside specific unknown classes.
        self.target_test = target_test + 1
        self.target_original = target_original + 1
        self.UUC_classes = np.array(UUC_classes) + 1
        self.col_labels = col_labels
        self.matrix = None
        self.title = title
        self.row_map = self.map_classes()

    def set_data(self, predict, target_test, target_original):
        self.predict = predict - self.predict_unknown_value
        self.target_test = target_test + 1
        self.target_original = target_original + 1

    def map_classes(self):
        """Maps each original class to its row; row 0 is reserved for unknown."""
        row_map = {}
        row_idx = 1   # row 0 is "unknown", so the known classes start at 1

        for c in np.unique(self.target_original):
            if c not in self.UUC_classes and c != 0:
                row_map[c] = row_idx
                row_idx += 1
            else:
                row_map[c] = 0
        return row_map

    def compute(self):
        if self.matrix is None:
            n_columns = len(np.unique(self.target_original))   # true classes
            n_rows = len(np.unique(self.target_test))          # predictions
            self.matrix = np.zeros((n_rows, n_columns))

        for predict, target_original in zip(self.predict, self.target_original):
            row = self.row_map[int(predict)]
            column = int(target_original)
            self.matrix[row][column] += 1

        return self.matrix

    def save(self, dir=None, name=None):
        if self.matrix is None:
            print("matrix has not been computed yet")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # White for empty cells, a blue ramp for everything else: normalizing
        # from vmin=1 leaves zero below the ramp, so it stays white.
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_lighter_blue", ["#FFFFFF", "royalblue"])
        norm = mcolors.Normalize(vmin=1, vmax=self.matrix.max())

        cax = ax.imshow(self.matrix, interpolation="nearest", cmap=cmap, norm=norm)
        fig.colorbar(cax)

        ax.set_title(f"Confusion Matrix - {self.title}" if self.title else "Confusion Matrix",
                     pad=20)

        row_labels = ["Unknown"] + [str(cls) for idx, cls in enumerate(self.col_labels)
                                    if (idx != 0 and idx not in self.UUC_classes)]

        ax.set_xticks(np.arange(len(self.col_labels)))
        ax.set_xticklabels(self.col_labels, rotation=45, ha="left")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)

        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

        # Counts are drawn in white on dark cells and black on light ones.
        threshold = self.matrix.max() / 2.
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                text_color = "white" if self.matrix[i, j] > threshold else "black"
                ax.text(j, i, str(int(self.matrix[i, j])),
                        ha="center", va="center", color=text_color)

        ax.set_xlabel("Real Class")
        ax.set_ylabel("Predicted Class")
        plt.tight_layout()

        if dir:
            os.makedirs(dir, exist_ok=True)
            plt.savefig(os.path.join(dir, f"confusion_matrix_{name}.png"))
        else:
            plt.savefig(f"confusion_matrix_{name}.png")

        plt.close()
