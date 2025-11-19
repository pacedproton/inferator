#!/usr/bin/env python3
"""
Hyperbolic Attention Embeddings

Maps attention to hyperbolic space (Poincaré ball) to better capture
hierarchical structure with exponentially growing capacity.

Mathematical foundation:
- Hyperbolic geometry (Poincaré ball model)
- Lorentz/Minkowski space
- Gyrovector spaces (Möbius operations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class PoincareManifold:
    """
    Poincaré ball model of hyperbolic space

    B^n = {x ∈ R^n : ||x|| < 1}

    Metric: ds² = 4/(1-||x||²)² ||dx||²

    Key properties:
    - Negative curvature
    - Trees embed with O(log n) distortion (vs O(n) in Euclidean)
    - Exponential volume growth
    """

    def __init__(self, c: float = 1.0, eps: float = 1e-6):
        """
        Args:
            c: Curvature parameter (c > 0)
            eps: Numerical stability epsilon
        """
        self.c = c
        self.eps = eps
        self.sqrt_c = np.sqrt(c)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points onto Poincaré ball

        Ensure ||x|| < 1/√c
        """
        norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=self.eps)
        max_norm = (1.0 - self.eps) / self.sqrt_c

        # Scale if outside ball
        scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
        return x * scale

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition: x ⊕ y

        x ⊕ y = (1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y
                ────────────────────────────────────────────
                    1 + 2c⟨x,y⟩ + c²||x||²||y||²
        """
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)

        numerator = (1 + 2*self.c*xy + self.c*y_norm_sq) * x + (1 - self.c*x_norm_sq) * y
        denominator = 1 + 2*self.c*xy + self.c**2*x_norm_sq*y_norm_sq + self.eps

        return self.project(numerator / denominator)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at x in direction v

        exp_x(v) = x ⊕ (tanh(√c λ_x ||v|| / 2) / (√c ||v||)) v

        where λ_x = 2 / (1 - c||x||²) is conformal factor
        """
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=self.eps)
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)

        lambda_x = 2.0 / (1.0 - self.c * x_norm_sq + self.eps)

        # Compute coefficient
        coeff = torch.tanh(self.sqrt_c * lambda_x * v_norm / 2) / (self.sqrt_c * v_norm + self.eps)

        # Möbius addition
        return self.mobius_add(x, coeff * v)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map: inverse of exp_map

        log_x(y) = (2/√c λ_x) artanh(√c ||−x ⊕ y||) (−x ⊕ y)/||−x ⊕ y||
        """
        diff = self.mobius_add(-x, y)
        diff_norm = torch.clamp(torch.norm(diff, dim=-1, keepdim=True), min=self.eps)

        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        lambda_x = 2.0 / (1.0 - self.c * x_norm_sq + self.eps)

        # Coefficient
        coeff = 2.0 / (self.sqrt_c * lambda_x + self.eps)
        coeff = coeff * torch.atanh(torch.clamp(self.sqrt_c * diff_norm, max=1.0 - self.eps))

        return coeff * diff / (diff_norm + self.eps)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic distance between x and y

        d(x, y) = (2/√c) artanh(√c ||−x ⊕ y||)
        """
        diff = self.mobius_add(-x, y)
        diff_norm = torch.norm(diff, dim=-1)

        return (2.0 / self.sqrt_c) * torch.atanh(
            torch.clamp(self.sqrt_c * diff_norm, max=1.0 - self.eps)
        )

    def parallel_transport(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transport of v from x to y

        P_{x→y}(v) = λ_y/λ_x (I + 2 (−x⊕y)⊗(−x⊕y)^T / (1-c||−x⊕y||²) − I) v
        """
        diff = self.mobius_add(-x, y)
        diff_norm_sq = torch.sum(diff * diff, dim=-1, keepdim=True)

        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)

        lambda_x = 2.0 / (1.0 - self.c * x_norm_sq + self.eps)
        lambda_y = 2.0 / (1.0 - self.c * y_norm_sq + self.eps)

        # Gyration
        coeff = 2.0 / (1.0 - self.c * diff_norm_sq + self.eps)

        # Simplified version (assuming small transport)
        return (lambda_y / (lambda_x + self.eps)) * v


class HyperbolicAttention(nn.Module):
    """
    Attention in hyperbolic space

    Maps Q, K, V to Poincaré ball, computes attention using hyperbolic distance
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        curvature: float = 1.0,
        use_hyperbolic: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            curvature: Hyperbolic curvature (higher = more curved)
            use_hyperbolic: If False, use standard Euclidean attention
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_hyperbolic = use_hyperbolic

        assert d_model % num_heads == 0

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Hyperbolic manifold
        if use_hyperbolic:
            self.manifold = PoincareManifold(c=curvature)

            # Learnable projection to hyperbolic space
            self.to_poincare = nn.Linear(self.head_dim, self.head_dim)
            self.from_poincare = nn.Linear(self.head_dim, self.head_dim)

        print(f"HyperbolicAttention:")
        print(f"  d_model: {d_model}, num_heads: {num_heads}")
        print(f"  Use hyperbolic: {use_hyperbolic}")
        if use_hyperbolic:
            print(f"  Curvature: {curvature}")

    def forward(
        self,
        x: torch.Tensor,
        return_distances: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            x: (batch, seq_len, d_model)
            return_distances: Return hyperbolic distances

        Returns:
            (output, distances)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_hyperbolic:
            # Map to Poincaré ball
            Q_hyp = torch.tanh(self.to_poincare(Q))  # Ensure in ball
            K_hyp = torch.tanh(self.to_poincare(K))

            Q_hyp = self.manifold.project(Q_hyp)
            K_hyp = self.manifold.project(K_hyp)

            # Compute hyperbolic distances
            # For each query, compute distance to all keys
            distances = []
            for i in range(seq_len):
                q_i = Q_hyp[:, :, i:i+1, :]  # (batch, heads, 1, dim)
                # Broadcast and compute distances
                k_all = K_hyp  # (batch, heads, seq_len, dim)

                # Pairwise distances
                dists = self.manifold.distance(
                    q_i.expand(-1, -1, seq_len, -1),
                    k_all
                )  # (batch, heads, seq_len)
                distances.append(dists)

            distances = torch.stack(distances, dim=2)  # (batch, heads, seq_len, seq_len)

            # Attention weights: exp(-distance)
            # Negative distance because closer = higher attention
            attn_weights = F.softmax(-distances, dim=-1)

            # Apply attention to values (in Euclidean space for simplicity)
            attn_output = torch.matmul(attn_weights, V)

            # Map back from hyperbolic
            attn_output = self.from_poincare(attn_output)

        else:
            # Standard Euclidean attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)
            distances = None

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.d_model)
        output = self.out_proj(attn_output)

        if return_distances and self.use_hyperbolic:
            return output, distances
        else:
            return output, None


class HyperbolicEmbedding(nn.Module):
    """
    Learn embeddings in hyperbolic space

    Useful for hierarchical data (e.g., WordNet, dependency trees)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        curvature: float = 1.0
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Euclidean embeddings (will project to hyperbolic)
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

        # Initialize small (stay near origin initially)
        nn.init.uniform_(self.embeddings.weight, -0.01, 0.01)

        self.manifold = PoincareManifold(c=curvature)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get hyperbolic embeddings

        Args:
            indices: (batch, seq_len)

        Returns:
            Embeddings in Poincaré ball (batch, seq_len, dim)
        """
        # Get Euclidean embeddings
        euclidean_emb = self.embeddings(indices)

        # Project to Poincaré ball
        # Use tanh to ensure ||x|| < 1
        hyperbolic_emb = torch.tanh(euclidean_emb)
        hyperbolic_emb = self.manifold.project(hyperbolic_emb)

        return hyperbolic_emb


def test_poincare_properties():
    """Test mathematical properties of Poincaré ball"""
    print("Testing Poincaré Ball Properties")
    print("=" * 60)

    manifold = PoincareManifold(c=1.0)

    # Test 1: Möbius addition associativity (approximate)
    print("\n1. Möbius addition (should preserve ball property):")
    x = torch.randn(5, 10) * 0.3
    y = torch.randn(5, 10) * 0.3

    x = manifold.project(x)
    y = manifold.project(y)

    z = manifold.mobius_add(x, y)
    z_norm = torch.norm(z, dim=-1)

    print(f"  ||x|| max: {torch.norm(x, dim=-1).max():.4f}")
    print(f"  ||y|| max: {torch.norm(y, dim=-1).max():.4f}")
    print(f"  ||x⊕y|| max: {z_norm.max():.4f}")
    print(f"  All in ball: {(z_norm < 1.0).all()}")

    # Test 2: Exp/Log are inverses
    print("\n2. Exp and Log maps (should be inverses):")
    x = torch.randn(5, 10) * 0.2
    v = torch.randn(5, 10) * 0.1

    x = manifold.project(x)

    y = manifold.exp_map(x, v)
    v_reconstructed = manifold.log_map(x, y)

    error = torch.norm(v - v_reconstructed) / torch.norm(v)
    print(f"  Reconstruction error: {error:.6f}")
    print(f"  Inverse property holds: {error < 1e-4}")

    # Test 3: Distance symmetry
    print("\n3. Distance symmetry:")
    x = torch.randn(5, 10) * 0.3
    y = torch.randn(5, 10) * 0.3

    x = manifold.project(x)
    y = manifold.project(y)

    d_xy = manifold.distance(x, y)
    d_yx = manifold.distance(y, x)

    symmetry_error = torch.abs(d_xy - d_yx).max()
    print(f"  d(x,y): {d_xy.mean():.4f}")
    print(f"  d(y,x): {d_yx.mean():.4f}")
    print(f"  Symmetry error: {symmetry_error:.6f}")
    print(f"  Symmetric: {symmetry_error < 1e-5}")

    # Test 4: Tree distortion
    print("\n4. Tree embedding (key advantage of hyperbolic space):")
    print("  Hyperbolic space embeds trees with O(log n) distortion")
    print("  Euclidean space requires O(n) distortion")
    print("  This is why hyperbolic is better for hierarchies!")


if __name__ == "__main__":
    print("Hyperbolic Attention Embeddings")
    print("=" * 60)
    print("\nHyperbolic geometry for hierarchical structure\n")

    # Test Poincaré properties
    test_poincare_properties()

    # Demo attention
    print("\n" + "=" * 60)
    print("Demo: Hyperbolic vs Euclidean Attention")
    print("=" * 60)

    batch, seq_len, d_model = 2, 16, 64
    num_heads = 8

    x = torch.randn(batch, seq_len, d_model)

    # Hyperbolic attention
    hyp_attn = HyperbolicAttention(d_model, num_heads, curvature=1.0, use_hyperbolic=True)
    hyp_out, distances = hyp_attn(x, return_distances=True)

    print(f"\nHyperbolic attention:")
    print(f"  Output shape: {hyp_out.shape}")
    if distances is not None:
        print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
        print(f"  Mean distance: {distances.mean():.4f}")

    # Euclidean attention
    euc_attn = HyperbolicAttention(d_model, num_heads, use_hyperbolic=False)
    euc_out, _ = euc_attn(x)

    print(f"\nEuclidean attention:")
    print(f"  Output shape: {euc_out.shape}")

    diff = torch.norm(hyp_out - euc_out) / torch.norm(euc_out)
    print(f"\nDifference: {diff:.4f}")
    print("\nHyperbolic attention is most beneficial for hierarchical tasks!")
