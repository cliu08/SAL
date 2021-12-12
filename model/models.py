
"""Script for Base Model."""
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self,
        in_channels: int = 3,
        hidden_channels: int = 10,
        out_channels: int = 20,
        cov_kernel_size: int = 5,
        maxpool_kernel_size: int = 2,
        classifier_in_features: int = 500,
        classifier_out_features: int = 50,
        num_labels: int = 10,
    ):
        """Initialization.

        Args:
            in_channels: the dimension of input space.
            hidden_channels: the dimension of hidden space.
            out_channels: the dimension of output space.
            cov_kernel_size: the number of kernels in convolutional
                layers.
            maxpool_kernel_size: the number of kernels in maxpooling
                layer.
            classifier_in_features: the dimension of inputs for
                classifier.
            classifier_out_features: the dimension of outputs for
                classifier.
            num_label: the number of labels.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.cov_kernel_size = cov_kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size
        self.classifier_in_features = classifier_in_features
        self.classifier_out_features = classifier_out_features
        self.num_labels = num_labels
        self.feature_extractor = self._feature_extractor()
        self.classifier = self._classifier()

    # @property
    # def feature_extractor(self):
    #     return self._feature_extractor()

    # @property
    # def classifier(self):
    #     return self._classifier()
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(3, 10, kernel_size=5),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(),
        #     nn.Conv2d(10, 20, kernel_size=5),
        #     nn.MaxPool2d(2),
        #     nn.Dropout2d(),
        # )
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(500, 50),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(50, num_labels),
        # )

    def _feature_extractor(self):
        """Extracting the features."""
        model = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.hidden_channels,
                kernel_size=self.cov_kernel_size,
            ),
            nn.MaxPool2d(self.maxpool_kernel_size),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_channels,
                self.out_channels,
                kernel_size=self.cov_kernel_size,
            ),
            nn.MaxPool2d(self.maxpool_kernel_size),
            nn.Dropout2d(),
        )
        return model

    def _classifier(self):
        """Classifier in the model."""
        model = nn.Sequential(
            nn.Linear(
                self.classifier_in_features,
                self.classifier_out_features,
            ),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(
                self.classifier_out_features,
                self.num_labels,
            ),
        )
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classification.

        Args:
            x: input data.

        Returns:
            Output labels.
        """
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits
