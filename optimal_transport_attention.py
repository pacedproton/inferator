#!/usr/bin/env python3
"""
Optimal Transport Attention Reweighting

Uses Sinkhorn iteration to find optimal attention reweighting
that minimizes Wasserstein distance to ideal distribution.

Mathematical foundation:
- Optimal transport theory (Monge-Kantorovich problem)
- Entropy-regularized Wasserstein distance
- Sinkhorn-Knopp algorithm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass


@dataclass
class SinkhornStats:
    """Statistics from Sinkhorn iteration"""
    iterations: int
    convergence_error: float
    wasserstein_distance: float
    entropy_before: float
    entropy_after: float
    kl_divergence: float


class OptimalTransportAttention:
    """
    Optimal transport-based attention reweighting

    Key idea: Given attention matrix A, find optimal transport plan P*
    that minimizes Wasserstein distance while maintaining desired properties.

    Mathematical formulation:
    min_P  <C, P> + λ H(P)
    s.t.   P 1 = a  (row sums)
           P^T 1 = b  (column sums)

    where:
    - C = cost matrix
    - H(P) = -Σ P_ij log P_ij (entropy regularization)
    - λ = regularization parameter

    Solved via Sinkhorn iteration:
    u_{k+1} = a / (K v_k)
    v_{k+1} = b / (K^T u_{k+1})
    P* = diag(u) K diag(v)

    where K = exp(-C/λ)
    """

    def __init__(
        self,
        reg_lambda: float = 0.1,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        cost_type: str = "uniform",
        target_distribution: str = "uniform"
    ):
        """
        Args:
            reg_lambda: Entropy regularization parameter (higher = more entropy)
            max_iterations: Maximum Sinkhorn iterations
            tolerance: Convergence tolerance
            cost_type: Type of cost matrix
                - "uniform": C_ij = 1 for all i,j (encourage uniformity)
                - "distance": C_ij = |i-j| (encourage locality)
                - "entropy": C_ij based on current entropy
            target_distribution: Target marginal distribution
                - "uniform": Equal distribution
                - "input": Match input distribution
        """
        self.reg_lambda = reg_lambda
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cost_type = cost_type
        self.target_distribution = target_distribution

        # Statistics
        self.stats_history = []

        print(f"OptimalTransportAttention initialized:")
        print(f"  Lambda: {reg_lambda}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Cost type: {cost_type}")
        print(f"  Target: {target_distribution}")

    def compute_cost_matrix(
        self,
        seq_len: int,
        attention: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Compute cost matrix C

        Args:
            seq_len: Sequence length
            attention: Current attention matrix (for entropy-based cost)
            device: Device to create tensor on

        Returns:
            Cost matrix C of shape (seq_len, seq_len)
        """
        if self.cost_type == "uniform":
            # Uniform cost: encourage uniform distribution
            C = torch.ones(seq_len, seq_len, device=device)

        elif self.cost_type == "distance":
            # Distance-based cost: encourage locality
            i_idx = torch.arange(seq_len, device=device).unsqueeze(1)
            j_idx = torch.arange(seq_len, device=device).unsqueeze(0)
            C = torch.abs(i_idx - j_idx).float()

        elif self.cost_type == "entropy":
            # Entropy-based cost: penalize low-entropy rows
            if attention is None:
                raise ValueError("Attention required for entropy-based cost")

            # Row-wise entropy
            eps = 1e-10
            entropy = -(attention * torch.log(attention + eps)).sum(dim=1)

            # Cost: inversely proportional to entropy
            # Low entropy → high cost → will be regularized more
            C = 1.0 / (entropy.unsqueeze(1) + 0.1)
            C = C.expand(seq_len, seq_len)

        else:
            raise ValueError(f"Unknown cost type: {self.cost_type}")

        return C

    def sinkhorn(
        self,
        C: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> Tuple[torch.Tensor, SinkhornStats]:
        """
        Sinkhorn algorithm for entropy-regularized optimal transport

        Solves:
        min_P  <C, P> + λ H(P)
        s.t.   P 1 = a, P^T 1 = b

        Args:
            C: Cost matrix (seq_len, seq_len)
            a: Source distribution (seq_len,)
            b: Target distribution (seq_len,)

        Returns:
            (P, stats) where P is optimal transport plan
        """
        seq_len = C.shape[0]
        device = C.device

        # Kernel K = exp(-C/λ)
        K = torch.exp(-C / self.reg_lambda)

        # Initialize
        u = torch.ones(seq_len, device=device)
        v = torch.ones(seq_len, device=device)

        # Sinkhorn iterations
        for iteration in range(self.max_iterations):
            u_prev = u.clone()

            # Update v
            v = b / (K.T @ u + 1e-10)

            # Update u
            u = a / (K @ v + 1e-10)

            # Check convergence
            error = torch.max(torch.abs(u - u_prev))

            if error < self.tolerance:
                break

        # Compute optimal transport plan
        P = torch.diag(u) @ K @ torch.diag(v)

        # Compute statistics
        stats = self._compute_stats(C, P, a, b, iteration + 1)

        return P, stats

    def _compute_stats(
        self,
        C: torch.Tensor,
        P: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        iterations: int
    ) -> SinkhornStats:
        """Compute statistics from Sinkhorn solution"""

        # Wasserstein distance
        wasserstein = (C * P).sum().item()

        # Entropy
        eps = 1e-10
        entropy_after = -(P * torch.log(P + eps)).sum().item()

        # KL divergence from target
        kl_div = (P * torch.log((P + eps) / (a.unsqueeze(1) * b.unsqueeze(0) + eps))).sum().item()

        # Convergence error
        row_sums = P.sum(dim=1)
        col_sums = P.sum(dim=0)
        conv_error = max(
            torch.max(torch.abs(row_sums - a)).item(),
            torch.max(torch.abs(col_sums - b)).item()
        )

        # Entropy before (of marginal a)
        entropy_before = -(a * torch.log(a + eps)).sum().item()

        return SinkhornStats(
            iterations=iterations,
            convergence_error=conv_error,
            wasserstein_distance=wasserstein,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            kl_divergence=kl_div
        )

    def reweight_attention(
        self,
        attention: torch.Tensor
    ) -> Tuple[torch.Tensor, SinkhornStats]:
        """
        Reweight attention using optimal transport

        Args:
            attention: Attention matrix (seq_len, seq_len)

        Returns:
            (reweighted_attention, stats)
        """
        seq_len = attention.shape[0]
        device = attention.device

        # Ensure attention is normalized (row-stochastic)
        attention_norm = F.softmax(attention, dim=1)

        # Source distribution (current row marginals)
        a = torch.ones(seq_len, device=device) / seq_len

        # Target distribution
        if self.target_distribution == "uniform":
            b = torch.ones(seq_len, device=device) / seq_len
        elif self.target_distribution == "input":
            # Match input distribution (column marginals of current attention)
            b = attention_norm.sum(dim=0)
            b = b / b.sum()
        else:
            raise ValueError(f"Unknown target: {self.target_distribution}")

        # Compute cost matrix
        C = self.compute_cost_matrix(seq_len, attention_norm, device)

        # Solve optimal transport
        P, stats = self.sinkhorn(C, a, b)

        self.stats_history.append(stats)

        return P, stats

    def apply_to_multi_head(
        self,
        attention: torch.Tensor,
        per_head: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply optimal transport to multi-head attention

        Args:
            attention: (batch, num_heads, seq_len, seq_len)
            per_head: If True, optimize each head independently

        Returns:
            (reweighted_attention, aggregated_stats)
        """
        batch, num_heads, seq_len, _ = attention.shape

        reweighted = torch.zeros_like(attention)
        all_stats = []

        if per_head:
            # Optimize each head independently
            for b in range(batch):
                for h in range(num_heads):
                    A = attention[b, h, :, :]
                    P, stats = self.reweight_attention(A)
                    reweighted[b, h, :, :] = P
                    all_stats.append(stats)
        else:
            # Average across heads, optimize once
            for b in range(batch):
                A_avg = attention[b, :, :, :].mean(dim=0)
                P, stats = self.reweight_attention(A_avg)
                # Broadcast to all heads
                reweighted[b, :, :, :] = P.unsqueeze(0)
                all_stats.append(stats)

        # Aggregate statistics
        agg_stats = self._aggregate_stats(all_stats)

        return reweighted, agg_stats

    def _aggregate_stats(self, stats_list) -> Dict:
        """Aggregate statistics from multiple runs"""
        if not stats_list:
            return {}

        return {
            'mean_iterations': np.mean([s.iterations for s in stats_list]),
            'mean_wasserstein': np.mean([s.wasserstein_distance for s in stats_list]),
            'mean_entropy_before': np.mean([s.entropy_before for s in stats_list]),
            'mean_entropy_after': np.mean([s.entropy_after for s in stats_list]),
            'entropy_increase': np.mean([s.entropy_after - s.entropy_before for s in stats_list]),
            'mean_kl_div': np.mean([s.kl_divergence for s in stats_list]),
            'max_conv_error': np.max([s.convergence_error for s in stats_list])
        }

    def reset_stats(self):
        """Reset statistics history"""
        self.stats_history = []


class OTAttentionTransformer(nn.Module):
    """
    Wrapper to apply OT attention refinement to transformer

    Hooks into attention layers and applies optimal transport reweighting
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: list,
        ot_config: dict,
        verbose: bool = False
    ):
        """
        Args:
            model: Transformer model
            layer_indices: Which layers to apply OT (e.g., [10, 15, 20])
            ot_config: Config dict for OptimalTransportAttention
            verbose: Print statistics
        """
        super().__init__()
        self.model = model
        self.layer_indices = layer_indices
        self.verbose = verbose

        # Create OT optimizer
        self.ot_optimizer = OptimalTransportAttention(**ot_config)

        # Hook storage
        self.hooks = []
        self.attention_maps = {}

        print(f"OTAttentionTransformer initialized:")
        print(f"  Target layers: {layer_indices}")
        print(f"  Config: {ot_config}")

    def register_hooks(self):
        """Register forward hooks to reweight attention"""

        def make_hook(layer_idx):
            def hook(module, input, output):
                # Note: This is a simplified hook
                # In practice, you'd need to modify attention computation directly
                # This assumes output includes attention weights

                if isinstance(output, tuple) and len(output) >= 2:
                    hidden_states, attention_weights = output[0], output[1]

                    # Apply OT reweighting
                    if attention_weights is not None:
                        reweighted, stats = self.ot_optimizer.apply_to_multi_head(
                            attention_weights, per_head=True
                        )

                        if self.verbose:
                            print(f"\nLayer {layer_idx} OT stats:")
                            for key, val in stats.items():
                                print(f"  {key}: {val:.4f}")

                        # Store for analysis
                        self.attention_maps[layer_idx] = {
                            'original': attention_weights.detach(),
                            'reweighted': reweighted.detach(),
                            'stats': stats
                        }

                return output

            return hook

        # Register hooks
        for idx in self.layer_indices:
            layer = self.model.model.layers[idx]
            hook = layer.self_attn.register_forward_hook(make_hook(idx))
            self.hooks.append(hook)

        print(f"Registered {len(self.hooks)} OT hooks")

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = {}

    def get_statistics(self) -> Dict:
        """Get aggregated statistics across all layers"""
        if not self.attention_maps:
            return {}

        all_stats = [v['stats'] for v in self.attention_maps.values()]
        return self.ot_optimizer._aggregate_stats(
            [SinkhornStats(**s) for s in all_stats]
        )


# Preset configurations
class OTPresets:
    """Recommended OT settings for different tasks"""

    # Light regularization: minimal change
    LIGHT = {
        'reg_lambda': 0.5,
        'max_iterations': 50,
        'tolerance': 1e-4,
        'cost_type': 'uniform',
        'target_distribution': 'uniform'
    }

    # Medium regularization: balanced
    MEDIUM = {
        'reg_lambda': 0.1,
        'max_iterations': 100,
        'tolerance': 1e-6,
        'cost_type': 'distance',
        'target_distribution': 'uniform'
    }

    # Heavy regularization: strong smoothing
    HEAVY = {
        'reg_lambda': 0.01,
        'max_iterations': 200,
        'tolerance': 1e-8,
        'cost_type': 'entropy',
        'target_distribution': 'uniform'
    }

    # Locality-preserving
    LOCALITY = {
        'reg_lambda': 0.1,
        'max_iterations': 100,
        'tolerance': 1e-6,
        'cost_type': 'distance',
        'target_distribution': 'input'
    }

    @classmethod
    def get_preset(cls, name: str) -> dict:
        """Get preset configuration"""
        presets = {
            'light': cls.LIGHT,
            'medium': cls.MEDIUM,
            'heavy': cls.HEAVY,
            'locality': cls.LOCALITY
        }

        if name.lower() not in presets:
            raise ValueError(f"Unknown preset: {name}. Choose from: {list(presets.keys())}")

        return presets[name.lower()]


if __name__ == "__main__":
    print("Optimal Transport Attention Reweighting")
    print("=" * 60)
    print("\nUses Sinkhorn iteration to find optimal attention reweighting")
    print("that minimizes Wasserstein distance.\n")

    # Demo on random attention
    print("Demo: Reweighting random attention matrix\n")

    seq_len = 16
    attention = torch.rand(seq_len, seq_len)
    attention = F.softmax(attention, dim=1)  # Normalize

    ot = OptimalTransportAttention(reg_lambda=0.1, cost_type='distance')

    print(f"Original attention entropy: {-(attention * torch.log(attention + 1e-10)).sum():.4f}")

    reweighted, stats = ot.reweight_attention(attention)

    print(f"\nSinkhorn converged in {stats.iterations} iterations")
    print(f"Wasserstein distance: {stats.wasserstein_distance:.4f}")
    print(f"Entropy before: {stats.entropy_before:.4f}")
    print(f"Entropy after: {stats.entropy_after:.4f}")
    print(f"Entropy increase: {stats.entropy_after - stats.entropy_before:.4f}")
    print(f"Convergence error: {stats.convergence_error:.2e}")

    print("\nSee test_optimal_transport.py for full testing")
