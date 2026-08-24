from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseCACClassifier(nn.Module, ABC):
    def __init__(self, num_classes: int, feat_dim: int, skip_distances: bool = False,
                 init_weights: bool = False):
        """
        Abstract base for Class Anchor Clustering (CAC) classifiers.

        Args:
            num_classes (int): number of classes in the problem.
            feat_dim (int): encoder output width (512 for ResNet18, 2048 for ResNet50).
            skip_distances (bool): if True, forward() skips the distance computation.
            init_weights (bool): if True, initialize the network weights.
        """
        super(BaseCACClassifier, self).__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.skip_distances = skip_distances

        # Built by the concrete subclass.
        self.encoder = self._build_encoder()

        # Projects the encoder features into class space.
        self.classify = nn.Linear(self.feat_dim, self.num_classes)

        # Anchor matrix, held fixed with respect to the gradient.
        self.anchors = nn.Parameter(
            torch.zeros(self.num_classes, self.num_classes).double(), requires_grad=False)

        if init_weights:
            self._initialize_weights()

    @abstractmethod
    def _build_encoder(self) -> nn.Module:
        """
        Required of every subclass: instantiate the backbone and replace its
        final classification layer with nn.Identity().
        """
        pass

    def forward(self, x):
        batch_size = len(x)

        x = self.encoder(x)
        x = x.view(batch_size, -1)

        outLinear = self.classify(x)

        if self.skip_distances:
            return outLinear

        outDistance = self.distance_classifier(outLinear)

        return outLinear, outDistance

    def distance_classifier(self, x):
        """Euclidean distance from x to each class anchor."""
        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = self.anchors.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x - anchors, 2, 2)

        return dists

    def set_anchors(self, means):
        """
            Sets the anchor centres, placing them on whichever device the rest
            of the model already lives on. That keeps this callable both before
            and after a .to(device) on the model.
        """
        device = next(self.parameters()).device
        self.anchors = nn.Parameter(means.double().to(device), requires_grad=False)

    def _initialize_weights(self):
        """Default weight initialization for the modules."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def save_model(self, dir: str):
        """Saves the weights and anchors to `dir`."""
        torch.save(self.state_dict(), dir)

    def predict_by_distance(self, epsilon, distances):
        """
            Turns anchor distances into predictions: the score of a class is
            its distance weighted by one minus the softmin over all distances,
            and a sample whose smallest score exceeds epsilon is rejected.
        """
        softmax = torch.nn.Softmax(dim=1)
        softmin = softmax(-distances)
        invScores = 1 - softmin
        scores = distances * invScores

        min_scores, predicted = torch.min(scores, axis=1)
        final_predictions = torch.where(min_scores > epsilon, -1, predicted)

        return final_predictions, min_scores, scores
