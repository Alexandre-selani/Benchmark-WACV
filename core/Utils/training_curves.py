"""Training/validation curves, written to disk at the end of a run."""

import os

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


class TrainingCurves:

    def __init__(self, name: str, dataset_name: str, dir: str = None):
        self.name = name
        self.dataset_name = dataset_name
        self.epochs = []

        if not dir:
            raise ValueError("TrainingCurves requires an output directory")
        self.dir = dir
        os.makedirs(dir, exist_ok=True)

        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def add_epoch(self,epoch:int=None,train_loss=None,train_acc=None,val_loss=None,val_acc=None):
        
        self.epochs.append(epoch)

        if train_loss:
            self.train_loss.append(train_loss)
        if train_acc:
            self.train_acc.append(train_acc)
        if val_loss:
            self.val_loss.append(val_loss)
        if val_acc:
            self.val_acc.append(val_acc)

    def save_plot(self):
        if not (self.train_loss and self.train_acc and self.val_acc and self.val_loss):
            return

        fig, ax1 = plt.subplots(figsize=(12, 8))

        # ── Loss (left axis) ──────────────────────────────────────────────────────
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color='tab:blue')
        ax1.plot(self.epochs, self.train_loss, color='tab:blue',   label='Train Loss')
        ax1.plot(self.epochs, self.val_loss,   color='tab:cyan',   label='Validation Loss', linestyle='--')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xticks(self.epochs)
        ax1.tick_params(axis='x', rotation=45)

        # ── Accuracy (right axis) ─────────────────────────────────────────────────
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", color='tab:red')
        ax2.plot(self.epochs, self.train_acc, color='tab:red',    label='Train Accuracy')
        ax2.plot(self.epochs, self.val_acc,   color='tab:orange', label='Validation Accuracy', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(0, 1)  # accuracy is always 0–1

        # ── Legend (merge both axes) ──────────────────────────────────────────────
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.title("Loss and Accuracy on Train/Validation")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, f"training_curves_{self.name}_{self.dataset_name}.png"))
        plt.close()
