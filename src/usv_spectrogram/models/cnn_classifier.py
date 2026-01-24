"""CNN classifier for USV detection."""

import torch
import torch.nn as nn
from typing import List


class USVClassifierCNN(nn.Module):
    """Small CNN for binary classification of USV spectrograms.

    Architecture:
        - 3 convolutional blocks (filters: [32, 64, 128])
        - Each block: Conv2d -> BatchNorm -> ReLU -> MaxPool
        - Global average pooling (handles variable input sizes)
        - Fully connected head with dropout
        - Output: single logit (use with BCEWithLogitsLoss)

    Args:
        num_filters: Number of filters in each conv layer (default: [32, 64, 128])
        dropout_rate: Dropout probability in classifier head (default: 0.5)
        optimal_threshold: Calibrated classification threshold (default: 0.40, with fixed padding to 512px)
    """

    def __init__(
        self,
        num_filters: List[int] = None,
        dropout_rate: float = 0.5,
        optimal_threshold: float = 0.40
    ):
        super().__init__()

        if num_filters is None:
            num_filters = [32, 64, 128]

        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.optimal_threshold = optimal_threshold

        # Convolutional feature extractor
        layers = []
        in_channels = 1  # Grayscale input

        for out_channels in num_filters:
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Global average pooling (handles variable input sizes)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head (outputs logits, NOT probabilities)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)  # Single logit output (NO sigmoid)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, H, W)

        Returns:
            Logits of shape (batch, 1) - use with BCEWithLogitsLoss
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions (for inference).

        Args:
            x: Input tensor of shape (batch, 1, H, W)

        Returns:
            Probabilities of shape (batch, 1) in range [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = None) -> torch.Tensor:
        """Get binary predictions (for inference).

        Args:
            x: Input tensor of shape (batch, 1, H, W)
            threshold: Decision threshold (default: use self.optimal_threshold)

        Returns:
            Binary predictions of shape (batch, 1)
        """
        if threshold is None:
            threshold = self.optimal_threshold
        proba = self.predict_proba(x)
        return (proba >= threshold).float()


class USVClassifierCNNLarge(nn.Module):
    """Large CNN for binary classification of USV spectrograms.

    NOT RECOMMENDED for small datasets (<5000 samples) - will overfit.

    Architecture:
        - 5 convolutional blocks (filters: [32, 64, 128, 256, 512])
        - Each block: Conv2d -> BatchNorm -> ReLU -> MaxPool
        - Global average pooling
        - Larger FC head (512 -> 128 -> 1)
        - Output: single logit

    Args:
        dropout_rate: Dropout probability in classifier head (default: 0.5)
        optimal_threshold: Calibrated classification threshold (default: 0.40, with fixed padding to 512px)
    """

    def __init__(self, dropout_rate: float = 0.5, optimal_threshold: float = 0.40):
        super().__init__()

        num_filters = [32, 64, 128, 256, 512]
        self.dropout_rate = dropout_rate
        self.optimal_threshold = optimal_threshold

        # Convolutional feature extractor
        layers = []
        in_channels = 1

        for out_channels in num_filters:
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Larger classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)  # NO sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = None) -> torch.Tensor:
        if threshold is None:
            threshold = self.optimal_threshold
        proba = self.predict_proba(x)
        return (proba >= threshold).float()
