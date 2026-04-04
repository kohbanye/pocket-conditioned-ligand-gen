"""EMA-updated vector quantization codebook shared by protein and ligand VQ-VAEs."""

import torch
from torch import Tensor, nn


class EMACodebook(nn.Module):
    """Exponential Moving Average codebook for vector quantization.

    Maintains a codebook of embedding vectors updated via EMA during training.
    Uses straight-through estimator for gradient propagation.
    Includes dead-code restart to prevent codebook collapse.
    """

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        ema_decay: float = 0.99,
        commitment_cost: float = 0.25,
        dead_code_threshold: int = 2,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.ema_decay = ema_decay
        self.commitment_cost = commitment_cost
        self.dead_code_threshold = dead_code_threshold

        embedding = torch.randn(num_codes, code_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embedding_sum", embedding.clone())
        self.register_buffer("usage_count", torch.zeros(num_codes, dtype=torch.long))

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize encoder output and compute commitment loss.

        Args:
            z: Encoder output of shape ``(B, code_dim)``.

        Returns:
            Tuple of (quantized, indices, commitment_loss).
        """
        # L2 nearest-neighbor lookup: (B, num_codes)
        distances = (
            z.pow(2).sum(dim=1, keepdim=True)
            - 2 * z @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = distances.argmin(dim=1)  # (B,)
        quantized = self.embedding[indices]  # (B, code_dim)

        if self.training:
            self._ema_update(z, indices)
            self._restart_dead_codes(z)

        # Commitment loss: encourages encoder output to stay close to codebook
        commitment_loss = self.commitment_cost * (z - quantized.detach()).pow(2).mean()

        # Straight-through estimator: copy gradients from quantized to z
        quantized = z + (quantized - z).detach()

        return quantized, indices, commitment_loss

    def _ema_update(self, z: Tensor, indices: Tensor) -> None:
        """Update codebook embeddings via exponential moving average."""
        one_hot = torch.zeros(
            indices.shape[0], self.num_codes, device=z.device,
        ).scatter_(1, indices.unsqueeze(1), 1.0)

        # Update cluster sizes
        batch_cluster_size = one_hot.sum(dim=0)
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            batch_cluster_size, alpha=1 - self.ema_decay,
        )

        # Update embedding sums
        batch_embedding_sum = one_hot.t() @ z
        self.ema_embedding_sum.mul_(self.ema_decay).add_(
            batch_embedding_sum, alpha=1 - self.ema_decay,
        )

        # Laplace smoothing to avoid division by zero
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + 1e-5) / (n + self.num_codes * 1e-5) * n

        self.embedding.copy_(self.ema_embedding_sum / smoothed.unsqueeze(1))

        # Track usage
        self.usage_count.add_((batch_cluster_size > 0).long())

    def _restart_dead_codes(self, z: Tensor) -> None:
        """Replace unused codebook entries with random encoder outputs."""
        dead_mask = self.usage_count < self.dead_code_threshold
        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return

        # Sample random encoder outputs as replacements
        num_samples = min(num_dead, z.shape[0])
        perm = torch.randperm(z.shape[0], device=z.device)[:num_samples]
        replacements = z[perm].detach()

        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:num_samples]
        self.embedding[dead_indices] = replacements
        self.ema_embedding_sum[dead_indices] = replacements
        self.ema_cluster_size[dead_indices] = 1.0
        self.usage_count[dead_indices] = 0

    def lookup(self, indices: Tensor) -> Tensor:
        """Look up codebook vectors by index (for decoding)."""
        return self.embedding[indices]
