"""Sequence-to-sequence reconstruction through a fixed-size bottleneck."""
import sys

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 latent_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=2,
                               batch_first=True, dropout=dropout)
        self.bottleneck = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.LSTM(latent_size, hidden_size, num_layers=2,
                               batch_first=True, dropout=dropout)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Windows ROCm MIOpen RNN kernels can fail HIPRTC compilation because
        # type_traits is missing. This flag also controls MIOpen dispatch on HIP.
        # Use native GPU LSTM ops; scope the flag so other models are unaffected.
        if sys.platform == "win32" and torch.version.hip is not None and x.is_cuda:
            previous = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False
            try:
                return self._reconstruct(x)
            finally:
                torch.backends.cudnn.enabled = previous
        return self._reconstruct(x)

    def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 150, sensors]. Only the final encoder state reaches decoder.
        _, (hidden, _) = self.encoder(x)
        latent = self.bottleneck(hidden[-1])
        repeated = latent.unsqueeze(1).expand(-1, x.size(1), -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def reconstruction_error(original: torch.Tensor,
                         reconstructed: torch.Tensor) -> torch.Tensor:
    """One MSE per window, averaged across time and standardized sensors."""
    return (original - reconstructed).square().mean(dim=(1, 2))


def reconstruction_loss(original: torch.Tensor, reconstructed: torch.Tensor,
                        delta: float = 1.0) -> torch.Tensor:
    """Mean Huber training/validation loss across batch, time, and sensors."""
    return nn.functional.huber_loss(reconstructed, original, reduction="mean", delta=delta)
