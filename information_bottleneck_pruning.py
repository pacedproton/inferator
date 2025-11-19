#!/usr/bin/env python3
"""
Information Bottleneck Pruning

Prune attention heads and neurons based on Information Bottleneck principle:
maximize I(Z; Y) while minimizing I(Z; X)

Mathematical foundation:
- Information Bottleneck (Tishby et al.)
- Mutual information estimation (MINE, KSG)
- Rate-distortion theory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class IBStats:
    """Information Bottleneck statistics"""
    mi_zx: float  # I(Z; X)
    mi_zy: float  # I(Z; Y)
    ib_objective: float  # I(Z; Y) - β I(Z; X)
    compression: float  # I(Z; X) / H(X)
    relevance: float  # I(Z; Y) / H(Y)


class MutualInformationEstimator:
    """
    Estimate mutual information using various methods

    I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)
    """

    def __init__(self, method: str = 'binning'):
        """
        Args:
            method: 'binning', 'ksg', or 'mine'
        """
        self.method = method

    def estimate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_bins: int = 10
    ) -> float:
        """
        Estimate I(X; Y)

        Args:
            x: (num_samples, dim_x)
            y: (num_samples, dim_y)
            num_bins: Number of bins for discretization

        Returns:
            MI estimate
        """
        if self.method == 'binning':
            return self._estimate_binning(x, y, num_bins)
        elif self.method == 'ksg':
            return self._estimate_ksg(x, y)
        elif self.method == 'mine':
            return self._estimate_mine(x, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _estimate_binning(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_bins: int
    ) -> float:
        """
        Estimate MI via binning/discretization

        Fast but less accurate for continuous variables
        """
        # Project to 1D if high-dimensional
        if x.dim() > 1 and x.shape[1] > 1:
            x = x @ torch.randn(x.shape[1], 1, device=x.device)
            x = x.squeeze()

        if y.dim() > 1 and y.shape[1] > 1:
            y = y @ torch.randn(y.shape[1], 1, device=y.device)
            y = y.squeeze()

        # Normalize to [0, num_bins)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_binned = ((x - x_min) / (x_max - x_min + 1e-10) * (num_bins - 1)).long()
        y_binned = ((y - y_min) / (y_max - y_min + 1e-10) * (num_bins - 1)).long()

        x_binned = torch.clamp(x_binned, 0, num_bins - 1)
        y_binned = torch.clamp(y_binned, 0, num_bins - 1)

        # Compute joint and marginal distributions
        joint = torch.zeros(num_bins, num_bins, device=x.device)
        for i in range(len(x)):
            joint[x_binned[i], y_binned[i]] += 1

        joint = joint / joint.sum()

        # Marginals
        px = joint.sum(dim=1)
        py = joint.sum(dim=0)

        # MI = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
        mi = 0.0
        for i in range(num_bins):
            for j in range(num_bins):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint[i, j] * torch.log(joint[i, j] / (px[i] * py[j]))

        return mi.item()

    def _estimate_ksg(self, x: torch.Tensor, y: torch.Tensor, k: int = 3) -> float:
        """
        Kraskov-Stögbauer-Grassberger (KSG) estimator

        Uses k-nearest neighbors
        """
        # Simplified KSG (full implementation requires scipy.spatial)
        # This is an approximation using correlation

        if x.dim() > 1:
            x = x.mean(dim=1)
        if y.dim() > 1:
            y = y.mean(dim=1)

        # Pearson correlation
        x_centered = x - x.mean()
        y_centered = y - y.mean()

        corr = (x_centered * y_centered).sum() / (
            torch.sqrt((x_centered**2).sum() * (y_centered**2).sum()) + 1e-10
        )

        # MI approximation: -0.5 log(1 - ρ²)
        mi = -0.5 * torch.log(1.0 - corr**2 + 1e-10)

        return max(0.0, mi.item())

    def _estimate_mine(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Mutual Information Neural Estimation (MINE)

        Requires training a neural network - simplified version here
        """
        # Simplified: use correlation-based approximation
        return self._estimate_ksg(x, y)


class InformationBottleneck:
    """
    Information Bottleneck framework for layer/head pruning

    Objective: max_θ I(Z; Y) - β I(Z; X)

    where:
    - Z = intermediate representation (head output, neuron activation)
    - X = input
    - Y = output/target
    - β = trade-off parameter
    """

    def __init__(
        self,
        beta: float = 1.0,
        mi_estimator: str = 'binning',
        num_bins: int = 10
    ):
        """
        Args:
            beta: IB trade-off parameter
            mi_estimator: Method for MI estimation
            num_bins: Number of bins for discretization
        """
        self.beta = beta
        self.mi_est = MutualInformationEstimator(method=mi_estimator)
        self.num_bins = num_bins

        print(f"InformationBottleneck:")
        print(f"  Beta: {beta}")
        print(f"  MI estimator: {mi_estimator}")

    def compute_ib_objective(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor
    ) -> IBStats:
        """
        Compute IB objective and statistics

        Args:
            x: Input (num_samples, dim_x)
            z: Intermediate representation (num_samples, dim_z)
            y: Output/target (num_samples, dim_y)

        Returns:
            IB statistics
        """
        # Estimate I(Z; X)
        mi_zx = self.mi_est.estimate(z, x, self.num_bins)

        # Estimate I(Z; Y)
        mi_zy = self.mi_est.estimate(z, y, self.num_bins)

        # IB objective
        ib_obj = mi_zy - self.beta * mi_zx

        # Entropies (for normalization)
        h_x = self._entropy(x)
        h_y = self._entropy(y)

        stats = IBStats(
            mi_zx=mi_zx,
            mi_zy=mi_zy,
            ib_objective=ib_obj,
            compression=mi_zx / (h_x + 1e-10),
            relevance=mi_zy / (h_y + 1e-10)
        )

        return stats

    def _entropy(self, x: torch.Tensor) -> float:
        """Estimate entropy via binning"""
        if x.dim() > 1 and x.shape[1] > 1:
            x = x @ torch.randn(x.shape[1], 1, device=x.device)
            x = x.squeeze()

        # Binning
        x_min, x_max = x.min(), x.max()
        x_binned = ((x - x_min) / (x_max - x_min + 1e-10) * (self.num_bins - 1)).long()
        x_binned = torch.clamp(x_binned, 0, self.num_bins - 1)

        # Distribution
        counts = torch.bincount(x_binned, minlength=self.num_bins).float()
        probs = counts / counts.sum()

        # Entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        return entropy.item()


class IBPruner:
    """
    Prune model components based on Information Bottleneck

    Identifies components with:
    - Low I(Z; Y): not relevant for output
    - High I(Z; X): not compressive

    Prunes components with low IB objective
    """

    def __init__(
        self,
        model: nn.Module,
        beta: float = 1.0,
        mi_estimator: str = 'binning'
    ):
        """
        Args:
            model: Neural network model
            beta: IB trade-off
            mi_estimator: MI estimation method
        """
        self.model = model
        self.ib = InformationBottleneck(beta, mi_estimator)

        # Statistics storage
        self.component_stats: Dict[str, IBStats] = {}
        self.pruning_mask: Dict[str, torch.Tensor] = {}

    def analyze_components(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target_layers: Optional[List[str]] = None
    ) -> Dict[str, IBStats]:
        """
        Analyze IB statistics for each component

        Args:
            x: Input (batch, ...)
            y: Target (batch, ...)
            target_layers: Which layers to analyze

        Returns:
            Dict of component name -> IB stats
        """
        # Forward pass with hooks to capture activations
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if target_layers is None or name in target_layers:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Forward pass
        with torch.no_grad():
            self.model(x)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Flatten inputs
        x_flat = x.view(x.shape[0], -1)
        y_flat = y.view(y.shape[0], -1) if isinstance(y, torch.Tensor) else y

        # Compute IB statistics for each component
        stats = {}
        for name, z in activations.items():
            z_flat = z.view(z.shape[0], -1)

            # Compute IB objective
            ib_stats = self.ib.compute_ib_objective(x_flat, z_flat, y_flat)
            stats[name] = ib_stats

        self.component_stats = stats
        return stats

    def compute_pruning_mask(
        self,
        threshold: Optional[float] = None,
        keep_ratio: float = 0.5
    ) -> Dict[str, bool]:
        """
        Compute pruning mask based on IB objective

        Args:
            threshold: IB objective threshold (keep if > threshold)
            keep_ratio: If threshold is None, keep this fraction

        Returns:
            Dict of component name -> keep (True/False)
        """
        if not self.component_stats:
            raise ValueError("Run analyze_components() first")

        # Get IB objectives
        objectives = {name: stats.ib_objective
                     for name, stats in self.component_stats.items()}

        # Determine threshold
        if threshold is None:
            # Keep top keep_ratio
            sorted_objs = sorted(objectives.values(), reverse=True)
            threshold = sorted_objs[int(len(sorted_objs) * keep_ratio)]

        # Create mask
        mask = {name: obj > threshold for name, obj in objectives.items()}

        self.pruning_mask = mask
        return mask

    def get_pruning_recommendations(self) -> List[Tuple[str, float, str]]:
        """
        Get pruning recommendations sorted by IB objective

        Returns:
            List of (component_name, ib_objective, recommendation)
        """
        recommendations = []

        for name, stats in sorted(
            self.component_stats.items(),
            key=lambda x: x[1].ib_objective
        ):
            if stats.ib_objective < 0:
                rec = "PRUNE (negative IB objective)"
            elif stats.relevance < 0.1:
                rec = "PRUNE (low relevance)"
            elif stats.compression > 0.9:
                rec = "PRUNE (high compression, info bottleneck)"
            else:
                rec = "KEEP"

            recommendations.append((name, stats.ib_objective, rec))

        return recommendations


def demo_ib_pruning():
    """Demonstrate IB-based pruning"""
    print("Information Bottleneck Pruning Demo")
    print("=" * 60)

    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 20)
            self.layer3 = nn.Linear(20, 5)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = self.layer3(x)
            return x

    model = SimpleModel()

    # Generate data
    x = torch.randn(100, 10)
    y = torch.randn(100, 5)

    print("\nAnalyzing model components with IB...")

    # Create pruner
    pruner = IBPruner(model, beta=1.0)

    # Analyze
    stats = pruner.analyze_components(x, y)

    print(f"\nComponent IB Statistics:")
    print(f"{'Component':<15} {'I(Z;X)':<10} {'I(Z;Y)':<10} {'IB Obj':<10} {'Compression':<12} {'Relevance':<10}")
    print("=" * 80)

    for name, s in stats.items():
        print(f"{name:<15} {s.mi_zx:<10.4f} {s.mi_zy:<10.4f} {s.ib_objective:<10.4f} {s.compression:<12.4f} {s.relevance:<10.4f}")

    # Get pruning recommendations
    print("\nPruning Recommendations:")
    print("=" * 60)

    recommendations = pruner.get_pruning_recommendations()
    for name, obj, rec in recommendations:
        print(f"{name:<15}: IB={obj:7.4f} -> {rec}")

    # Compute mask
    mask = pruner.compute_pruning_mask(keep_ratio=0.7)

    print("\nPruning Mask (keep_ratio=0.7):")
    for name, keep in mask.items():
        print(f"  {name}: {'KEEP' if keep else 'PRUNE'}")


if __name__ == "__main__":
    print("Information Bottleneck Pruning")
    print("=" * 60)
    print("\nPrinciple: Keep components that are relevant to output")
    print("while compressing information from input\n")

    demo_ib_pruning()

    print("\n" + "=" * 60)
    print("Key Insights:")
    print("  - High I(Z; Y): relevant for output")
    print("  - Low I(Z; X): compressed representation")
    print("  - IB objective = I(Z; Y) - β I(Z; X)")
    print("  - Prune components with low IB objective")
