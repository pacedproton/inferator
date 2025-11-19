#!/usr/bin/env python3
"""
Harmonic Attention Filter

Spectral denoising of attention values using Graph Fourier Transform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class FilteringState:
    """State of harmonic filtering at one layer"""
    layer_idx: int
    energy_before: float
    energy_after: float
    reduction_ratio: float
    num_high_freq: int
    spectral_gap: float


class HarmonicAttentionFilter:
    """
    Spectral filter for attention values using Graph Laplacian

    Key idea: Treat attention matrix A as adjacency of token graph,
    then filter value vectors V using graph Fourier transform.

    Mathematical foundation:
    - Graph Laplacian: L = D - A (D is degree matrix)
    - Dirichlet energy: E(v) = v^T L v (measures signal roughness)
    - Filtering: v_filtered = (I + γL)^{-1} v (heat equation solution)
    """

    def __init__(
        self,
        gamma: float = 0.1,
        method: str = "tikhonov",
        cutoff_ratio: float = 0.3,
        window_size: Optional[int] = None,
        detect_shifts: bool = False
    ):
        """
        Args:
            gamma: Regularization strength (higher = more smoothing)
            method: "tikhonov" or "spectral"
                - tikhonov: (I + γL)^{-1} V (implicit filtering)
                - spectral: Explicit eigendecomposition + cutoff
            cutoff_ratio: For spectral method, fraction of eigenvalues to keep
            window_size: If set, use sliding window (memory efficient)
            detect_shifts: Use Fiedler vector for context boundary detection
        """
        self.gamma = gamma
        self.method = method
        self.cutoff_ratio = cutoff_ratio
        self.window_size = window_size
        self.detect_shifts = detect_shifts

        # Statistics
        self.history: List[FilteringState] = []
        self.shift_points: List[int] = []

        print(f"HarmonicAttentionFilter initialized:")
        print(f"  Method: {method}")
        print(f"  Gamma: {gamma}")
        if method == "spectral":
            print(f"  Cutoff ratio: {cutoff_ratio}")
        if window_size:
            print(f"  Window size: {window_size}")

    def compute_laplacian(
        self,
        attention_weights: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute graph Laplacian from attention weights

        Args:
            attention_weights: (seq_len, seq_len) attention matrix
            normalize: If True, use normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2}

        Returns:
            Laplacian matrix (seq_len, seq_len)
        """
        # Symmetrize attention (undirected graph)
        A = (attention_weights + attention_weights.T) / 2.0

        # Degree matrix
        degrees = A.sum(dim=1)
        D = torch.diag(degrees)

        if normalize:
            # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees + 1e-10))
            L = torch.eye(A.shape[0], device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
        else:
            # Combinatorial Laplacian: D - A
            L = D - A

        return L

    def compute_dirichlet_energy(
        self,
        v: torch.Tensor,
        L: torch.Tensor
    ) -> float:
        """
        Compute Dirichlet energy: E(v) = v^T L v

        Measures signal roughness - high energy = noisy/oscillating signal

        Args:
            v: Signal vector (seq_len, d) or (seq_len,)
            L: Laplacian (seq_len, seq_len)

        Returns:
            Energy (scalar)
        """
        if v.dim() == 1:
            # Single vector
            energy = (v @ L @ v).item()
        else:
            # Multiple vectors (seq_len, d)
            # E = tr(V^T L V) = sum_i v_i^T L v_i
            energy = torch.trace(v.T @ L @ v).item()

        return energy

    def tikhonov_filter(
        self,
        V: torch.Tensor,
        L: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
        """
        Tikhonov regularization filter: Ṽ = (I + γL)^{-1} V

        Solves: argmin_Ṽ ||Ṽ - V||^2 + γ·E(Ṽ)

        Equivalent to solving heat equation on graph:
        dV/dt = -L V with V(0) = V_0

        Args:
            V: Value matrix (seq_len, d)
            L: Laplacian (seq_len, seq_len)
            gamma: Regularization strength

        Returns:
            Filtered values Ṽ (seq_len, d)
        """
        seq_len = V.shape[0]
        I = torch.eye(seq_len, device=V.device)

        # Solve (I + γL) Ṽ = V
        # Using torch.linalg.solve for numerical stability
        M = I + gamma * L
        V_filtered = torch.linalg.solve(M, V)

        return V_filtered

    def spectral_filter(
        self,
        V: torch.Tensor,
        L: torch.Tensor,
        cutoff_ratio: float
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Spectral filtering via eigendecomposition

        Steps:
        1. Decompose L = U Λ U^T
        2. Transform V to frequency domain: V̂ = U^T V
        3. Zero out high frequencies (large eigenvalues)
        4. Inverse transform: Ṽ = U V̂_filtered

        Args:
            V: Value matrix (seq_len, d)
            L: Laplacian (seq_len, seq_len)
            cutoff_ratio: Keep this fraction of eigenvalues

        Returns:
            (filtered_V, num_high_freq_zeroed, spectral_gap)
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        # Determine cutoff
        k = int(cutoff_ratio * len(eigenvalues))
        cutoff_idx = k

        # Create filter
        filter_mask = torch.zeros_like(eigenvalues)
        filter_mask[:cutoff_idx] = 1.0

        # Transform to frequency domain
        V_hat = eigenvectors.T @ V  # (seq_len, d)

        # Apply filter
        V_hat_filtered = V_hat * filter_mask.unsqueeze(1)

        # Inverse transform
        V_filtered = eigenvectors @ V_hat_filtered

        # Compute spectral gap (between kept and removed)
        if cutoff_idx < len(eigenvalues):
            spectral_gap = (eigenvalues[cutoff_idx] - eigenvalues[cutoff_idx-1]).item()
        else:
            spectral_gap = 0.0

        num_high_freq = len(eigenvalues) - cutoff_idx

        return V_filtered, num_high_freq, spectral_gap

    def filter_values(
        self,
        values: torch.Tensor,
        attention_weights: torch.Tensor,
        layer_idx: int = 0
    ) -> Tuple[torch.Tensor, FilteringState]:
        """
        Main filtering function

        Args:
            values: (batch, num_heads, seq_len, head_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len)
            layer_idx: Layer index for logging

        Returns:
            (filtered_values, state)
        """
        batch, num_heads, seq_len, head_dim = values.shape

        # Process each head independently
        filtered_values = torch.zeros_like(values)

        total_energy_before = 0.0
        total_energy_after = 0.0
        total_high_freq = 0
        total_spectral_gap = 0.0

        for b in range(batch):
            for h in range(num_heads):
                # Extract single head
                V = values[b, h, :, :]  # (seq_len, head_dim)
                A = attention_weights[b, h, :, :]  # (seq_len, seq_len)

                # Sliding window if specified
                if self.window_size and seq_len > self.window_size:
                    V_filtered = self._filter_with_sliding_window(V, A)
                else:
                    # Compute Laplacian
                    L = self.compute_laplacian(A, normalize=True)

                    # Measure before
                    energy_before = self.compute_dirichlet_energy(V, L)
                    total_energy_before += energy_before

                    # Apply filter
                    if self.method == "tikhonov":
                        V_filtered = self.tikhonov_filter(V, L, self.gamma)
                        num_high_freq = 0
                        spectral_gap = 0.0
                    else:  # spectral
                        V_filtered, num_high_freq, spectral_gap = self.spectral_filter(
                            V, L, self.cutoff_ratio
                        )
                        total_high_freq += num_high_freq
                        total_spectral_gap += spectral_gap

                    # Measure after
                    energy_after = self.compute_dirichlet_energy(V_filtered, L)
                    total_energy_after += energy_after

                filtered_values[b, h, :, :] = V_filtered

        # Average across batch and heads
        num_total = batch * num_heads
        avg_energy_before = total_energy_before / num_total
        avg_energy_after = total_energy_after / num_total
        reduction_ratio = 1.0 - (avg_energy_after / (avg_energy_before + 1e-10))

        # Create state
        state = FilteringState(
            layer_idx=layer_idx,
            energy_before=avg_energy_before,
            energy_after=avg_energy_after,
            reduction_ratio=reduction_ratio,
            num_high_freq=total_high_freq // num_total if self.method == "spectral" else 0,
            spectral_gap=total_spectral_gap / num_total if self.method == "spectral" else 0.0
        )

        self.history.append(state)

        return filtered_values, state

    def _filter_with_sliding_window(
        self,
        V: torch.Tensor,
        A: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply filtering with sliding window for long sequences

        Args:
            V: (seq_len, head_dim)
            A: (seq_len, seq_len)

        Returns:
            Filtered V
        """
        seq_len = V.shape[0]
        window_size = self.window_size
        stride = window_size // 2  # 50% overlap

        V_filtered = torch.zeros_like(V)
        counts = torch.zeros(seq_len, device=V.device)

        for start in range(0, seq_len, stride):
            end = min(start + window_size, seq_len)

            # Extract window
            V_window = V[start:end, :]
            A_window = A[start:end, start:end]

            # Compute Laplacian
            L_window = self.compute_laplacian(A_window, normalize=True)

            # Filter
            if self.method == "tikhonov":
                V_window_filtered = self.tikhonov_filter(V_window, L_window, self.gamma)
            else:
                V_window_filtered, _, _ = self.spectral_filter(
                    V_window, L_window, self.cutoff_ratio
                )

            # Accumulate (for overlapping regions)
            V_filtered[start:end, :] += V_window_filtered
            counts[start:end] += 1

        # Average overlapping regions
        V_filtered = V_filtered / counts.unsqueeze(1)

        return V_filtered

    def detect_context_shift(
        self,
        attention_weights: torch.Tensor,
        threshold: float = 0.1
    ) -> List[int]:
        """
        Detect context boundaries using Fiedler vector (2nd eigenvector of L)

        The Fiedler vector provides a natural graph partitioning:
        sign changes indicate community boundaries.

        Args:
            attention_weights: (seq_len, seq_len)
            threshold: Minimum change to consider a boundary

        Returns:
            List of boundary positions
        """
        # Compute Laplacian
        L = self.compute_laplacian(attention_weights, normalize=True)

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        # Fiedler vector (2nd smallest eigenvalue)
        fiedler = eigenvectors[:, 1].cpu().numpy()

        # Detect sign changes and large jumps
        boundaries = []
        for i in range(1, len(fiedler)):
            # Sign change
            if fiedler[i] * fiedler[i-1] < 0:
                boundaries.append(i)
            # Large magnitude change
            elif abs(fiedler[i] - fiedler[i-1]) > threshold:
                boundaries.append(i)

        self.shift_points = boundaries
        return boundaries

    def get_statistics(self) -> Dict:
        """Get summary statistics from filtering history"""
        if not self.history:
            return {}

        energies_before = [s.energy_before for s in self.history]
        energies_after = [s.energy_after for s in self.history]
        reductions = [s.reduction_ratio for s in self.history]

        return {
            'num_filtered': len(self.history),
            'mean_energy_before': float(np.mean(energies_before)),
            'mean_energy_after': float(np.mean(energies_after)),
            'mean_reduction': float(np.mean(reductions)),
            'total_reduction': float(np.sum(reductions)),
            'energy_range_before': (float(np.min(energies_before)), float(np.max(energies_before))),
            'energy_range_after': (float(np.min(energies_after)), float(np.max(energies_after))),
            'gamma': self.gamma,
            'method': self.method,
        }

    def reset(self):
        """Reset statistics"""
        self.history = []
        self.shift_points = []


class HarmonicTransformer(nn.Module):
    """
    Wrapper for applying harmonic filtering to transformer layers

    Hooks into specified layers and filters attention values in-place
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: List[int],
        filter_config: Dict,
        verbose: bool = False
    ):
        """
        Args:
            model: Transformer model
            layer_indices: Which layers to filter (e.g., [10, 15, 20])
            filter_config: Config dict for HarmonicAttentionFilter
            verbose: Print filtering statistics
        """
        super().__init__()
        self.model = model
        self.layer_indices = layer_indices
        self.verbose = verbose

        # Create filter
        self.filter = HarmonicAttentionFilter(**filter_config)

        # Storage for hooks
        self.hooks = []
        self.attention_outputs = {}

        print(f"HarmonicTransformer initialized:")
        print(f"  Filtering layers: {layer_indices}")
        print(f"  Config: {filter_config}")

    def _get_hook(self, layer_idx: int) -> Callable:
        """Create hook function for a specific layer"""

        def hook(module, input, output):
            """
            Hook for attention layer

            Note: This assumes HuggingFace transformer structure where
            output is (hidden_states, attention_weights) tuple
            """
            # Extract attention outputs
            # For HuggingFace models with output_attentions=True
            if isinstance(output, tuple) and len(output) >= 2:
                hidden_states = output[0]
                attention_weights = output[1]

                # Store for potential analysis
                self.attention_outputs[layer_idx] = attention_weights

                # Note: We can't directly filter here without modifying
                # the attention mechanism. This is a simplified version.
                # In practice, you'd need to hook into the attention
                # computation before the output projection.

                if self.verbose:
                    print(f"Layer {layer_idx}: captured attention of shape {attention_weights.shape}")

            return output

        return hook

    def register_hooks(self):
        """Register forward hooks on target layers"""
        for idx in self.layer_indices:
            layer = self.model.model.layers[idx]
            hook = layer.self_attn.register_forward_hook(self._get_hook(idx))
            self.hooks.append(hook)

        print(f"Registered {len(self.hooks)} hooks")

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_outputs = {}

    def forward(self, *args, **kwargs):
        """Forward pass through model with filtering"""
        return self.model(*args, **kwargs)


# Preset configurations
class FilterPresets:
    """Recommended filtering settings for different tasks"""

    # Light smoothing - preserves most details
    LIGHT = {
        'gamma': 0.05,
        'method': 'tikhonov',
        'window_size': None,
        'detect_shifts': False
    }

    # Medium smoothing - balanced
    MEDIUM = {
        'gamma': 0.1,
        'method': 'tikhonov',
        'window_size': None,
        'detect_shifts': False
    }

    # Heavy smoothing - aggressive denoising
    HEAVY = {
        'gamma': 0.3,
        'method': 'tikhonov',
        'window_size': None,
        'detect_shifts': False
    }

    # Spectral filtering - explicit frequency cutoff
    SPECTRAL = {
        'gamma': 0.1,
        'method': 'spectral',
        'cutoff_ratio': 0.3,
        'window_size': None,
        'detect_shifts': False
    }

    # Long context - sliding window
    LONG_CONTEXT = {
        'gamma': 0.1,
        'method': 'tikhonov',
        'window_size': 512,
        'detect_shifts': True
    }

    @classmethod
    def get_preset(cls, name: str) -> Dict:
        """Get preset configuration"""
        presets = {
            'light': cls.LIGHT,
            'medium': cls.MEDIUM,
            'heavy': cls.HEAVY,
            'spectral': cls.SPECTRAL,
            'long': cls.LONG_CONTEXT
        }

        if name.lower() not in presets:
            raise ValueError(f"Unknown preset: {name}. Choose from: {list(presets.keys())}")

        return presets[name.lower()]


if __name__ == "__main__":
    print("Harmonic Attention Filter - Spectral Denoising for Transformers")
    print("="*60)
    print("\nThis module provides graph-theoretic filtering of attention values")
    print("using the Graph Laplacian and spectral methods.")
    print("\nSee test_harmonic.py for usage examples.")
