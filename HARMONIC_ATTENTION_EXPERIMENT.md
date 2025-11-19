# Harmonic Attention Filtering: Spectral Denoising via Graph Fourier Transform

## The Revolutionary Idea

**Apply spectral graph theory to denoise attention, filtering high-frequency "semantic jitter" while preserving low-frequency coherent structure.**

Standard attention:
- Values weighted by attention matrix
- No smoothness constraint
- Susceptible to high-frequency noise (hallucinations)

**Harmonic Attention**:
- Values filtered through graph Laplacian
- Enforces smoothness on attention graph
- Removes semantic outliers (noise, hallucinations)
- Based on heat equation / diffusion on graphs

## Mathematical Foundation

### The Problem: Semantic Jitter

Attention defines a **graph** where:
- Nodes = tokens
- Edges = attention weights A_ij
- Signal = value vectors V

**Problem**: When hallucinating, V contains **high-frequency components** - abrupt changes between semantically similar tokens.

**Analogy**:
- Low freq = coherent theme (melody)
- High freq = local noise (static)
- Hallucination = signal with too much high freq

### The Solution: Spectral Filtering

Filter V using the **graph Laplacian** derived from attention matrix A.

### Graph Laplacian

**Symmetrized attention** (undirected graph):
```
W = (A + A^T) / 2
```

**Degree matrix**:
```
D_ii = ∑_j W_ij
```

**Combinatorial Laplacian**:
```
L = D - W
```

**Properties**:
- L is positive semi-definite
- L1 = 0 (constant vector is harmonic)
- x^T L x measures "smoothness" of signal x

### Spectral Decomposition

**Eigendecomposition** of L:
```
L = U Λ U^T
```

Where:
- **U**: Eigenvectors (harmonic modes / graph frequencies)
- **Λ**: Eigenvalues (frequencies)
  - λ_0 = 0: DC component (global average)
  - Small λ: Low freq (smooth, coherent)
  - Large λ: High freq (oscillatory, noisy)

**Graph Fourier Transform**:
```
V̂ = U^T V  (forward - decompose into frequencies)
Ṽ = U V̂    (inverse - reconstruct from frequencies)
```

### Tikhonov Regularization on Graphs

**Optimization problem**: Find filtered signal Ṽ that is:
1. Close to original V (fidelity)
2. Smooth on the graph (low Dirichlet energy)

```
minimize: ||Ṽ - V||² + γ · Ṽ^T L Ṽ
```

**Closed-form solution**:
```
Ṽ = (I + γL)^{-1} V
```

**Interpretation**: Heat equation diffusion
- γ: Diffusion time / smoothing strength
- (I + γL)^{-1}: Heat kernel approximation
- Diffuses "hot spots" (inconsistent values) to neighbors

### Dirichlet Energy

Measures "roughness" of signal on graph:

```
E(V) = V^T L V = ∑_{i,j} W_ij (V_i - V_j)²
```

**Interpretation**:
- High energy: Values change abruptly between connected tokens
- Low energy: Values are smooth across attention graph

**Hypothesis**: Hallucinations have high Dirichlet energy (semantic inconsistency).

### Connection to Heat Equation

The filter (I + γL)^{-1} approximates solving:

```
∂u/∂t = -Δ_G u   (heat equation on graph G)

u(t=γ) ≈ (I + γL)^{-1} u(0)
```

**Physical interpretation**: Truth is thermodynamic equilibrium. Hallucinations are "hot spots" that diffuse away.

## Implementation

### Core Harmonic Filter

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import numpy as np

class HarmonicAttentionFilter:
    """
    Spectral filter for attention values using graph Laplacian

    Applies low-pass filtering to remove high-frequency semantic noise
    while preserving coherent low-frequency structure.
    """

    def __init__(
        self,
        window_size: int = 32,
        gamma: float = 0.5,
        method: str = 'tikhonov',
        num_modes: Optional[int] = None
    ):
        """
        Args:
            window_size: Size of local attention window
            gamma: Smoothing strength (regularization parameter)
                - gamma → 0: No smoothing
                - gamma → ∞: Over-smoothing (constant signal)
            method: Filtering method
                - 'tikhonov': (I + γL)^{-1} - fast, implicit
                - 'spectral': U Λ̂ U^T - explicit frequency control
            num_modes: For spectral method, number of low-freq modes to keep
        """
        self.window_size = window_size
        self.gamma = gamma
        self.method = method
        self.num_modes = num_modes

        # Statistics
        self.energy_reduction = []
        self.filter_norms = []

    def compute_laplacian(
        self,
        attention_weights: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute graph Laplacian from attention matrix

        Args:
            attention_weights: (batch, heads, window, window)
            normalize: Use normalized Laplacian L_norm = D^{-1/2} L D^{-1/2}

        Returns:
            Laplacian tensor
        """
        # Symmetrize attention (undirected graph approximation)
        # W = (A + A^T) / 2
        adj = 0.5 * (attention_weights + attention_weights.transpose(-1, -2))

        # Degree matrix
        # D_ii = sum_j W_ij
        degree = torch.sum(adj, dim=-1)  # (batch, heads, window)

        # Combinatorial Laplacian: L = D - W
        degree_mat = torch.diag_embed(degree)  # (batch, heads, window, window)
        laplacian = degree_mat - adj

        if normalize:
            # Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
            # More stable for graph signal processing
            degree_inv_sqrt = torch.pow(degree + 1e-10, -0.5)
            degree_inv_sqrt_mat = torch.diag_embed(degree_inv_sqrt)

            laplacian = degree_inv_sqrt_mat @ laplacian @ degree_inv_sqrt_mat

        return laplacian

    def compute_dirichlet_energy(
        self,
        values: torch.Tensor,
        laplacian: torch.Tensor
    ) -> float:
        """
        Compute Dirichlet energy: E(V) = V^T L V

        Measures "roughness" of values on attention graph

        Args:
            values: (batch, heads, window, dim)
            laplacian: (batch, heads, window, window)

        Returns:
            Average Dirichlet energy
        """
        B, H, W, D = values.shape

        # Compute V^T L V for each dimension
        energy = 0.0
        for d in range(D):
            v = values[:, :, :, d]  # (B, H, W)
            # v^T L v
            lv = torch.matmul(laplacian, v.unsqueeze(-1)).squeeze(-1)
            energy += (v * lv).sum()

        # Average over all elements
        return (energy / (B * H * W * D)).item()

    def tikhonov_filter(
        self,
        values: torch.Tensor,
        laplacian: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Tikhonov regularization filter: (I + γL)^{-1} V

        Args:
            values: (batch, heads, window, dim)
            laplacian: (batch, heads, window, window)

        Returns:
            Filtered values
        """
        B, H, W, D = values.shape

        # Construct filter matrix: I + γL
        eye = torch.eye(W, device=values.device).view(1, 1, W, W)
        filter_matrix = eye + self.gamma * laplacian

        # Solve linear system: (I + γL) x = V
        # This is more stable than computing inverse directly

        # Flatten batch and heads for efficient solving
        filter_flat = filter_matrix.reshape(B * H, W, W)
        values_flat = values.reshape(B * H, W, D)

        # Solve Ax = b for each dimension
        filtered_flat = torch.linalg.solve(filter_flat, values_flat)

        # Reshape back
        filtered = filtered_flat.reshape(B, H, W, D)

        return filtered

    def spectral_filter(
        self,
        values: torch.Tensor,
        laplacian: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply spectral filter: U Λ̂ U^T V

        Explicitly zero out high-frequency components

        Args:
            values: (batch, heads, window, dim)
            laplacian: (batch, heads, window, window)

        Returns:
            Filtered values
        """
        B, H, W, D = values.shape

        # Eigendecomposition: L = U Λ U^T
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        # eigenvalues: (B, H, W) - sorted ascending (λ_0 = 0, λ_1, ...)
        # eigenvectors: (B, H, W, W) - columns are eigenvectors

        # Determine cutoff frequency
        if self.num_modes is None:
            # Keep frequencies below threshold
            # λ / λ_max < cutoff
            lambda_max = eigenvalues[:, :, -1].unsqueeze(-1)  # (B, H, 1)
            cutoff_ratio = 0.3  # Keep bottom 30% of spectrum
            mask = (eigenvalues / (lambda_max + 1e-10)) < cutoff_ratio
        else:
            # Keep first num_modes eigenvectors
            mask = torch.zeros_like(eigenvalues, dtype=torch.bool)
            mask[:, :, :self.num_modes] = True

        # Create filtered eigenvalue matrix
        lambda_filtered = torch.where(mask, eigenvalues, torch.zeros_like(eigenvalues))
        lambda_mat = torch.diag_embed(lambda_filtered)

        # Apply filter: V_filtered = U (I - Λ̂) U^T V
        # Or: V_filtered = V - U Λ̂ U^T V
        # This removes high-frequency components

        # Transform values to spectral domain
        # V̂ = U^T V (for each dimension)
        filtered = []
        for d in range(D):
            v = values[:, :, :, d].unsqueeze(-1)  # (B, H, W, 1)

            # Forward transform: v̂ = U^T v
            v_hat = torch.matmul(eigenvectors.transpose(-1, -2), v)

            # Apply filter: zero high frequencies
            v_hat_filtered = torch.where(
                mask.unsqueeze(-1),
                v_hat,
                torch.zeros_like(v_hat)
            )

            # Inverse transform: v_filtered = U v̂_filtered
            v_filtered = torch.matmul(eigenvectors, v_hat_filtered).squeeze(-1)

            filtered.append(v_filtered)

        filtered = torch.stack(filtered, dim=-1)  # (B, H, W, D)

        return filtered

    def filter_values(
        self,
        attention_weights: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Main filtering function

        Args:
            attention_weights: (batch, heads, seq_len, seq_len)
            values: (batch, heads, seq_len, dim)

        Returns:
            (filtered_values, statistics)
        """
        B, H, S, D = values.shape

        # Process in sliding windows for efficiency
        if S <= self.window_size:
            # Small enough, process all at once
            laplacian = self.compute_laplacian(attention_weights)

            # Compute energy before filtering
            energy_before = self.compute_dirichlet_energy(values, laplacian)

            # Apply filter
            if self.method == 'tikhonov':
                filtered = self.tikhonov_filter(values, laplacian)
            elif self.method == 'spectral':
                filtered = self.spectral_filter(values, laplacian)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Compute energy after filtering
            energy_after = self.compute_dirichlet_energy(filtered, laplacian)

            stats = {
                'energy_before': energy_before,
                'energy_after': energy_after,
                'energy_reduction': energy_before - energy_after,
                'reduction_ratio': (energy_before - energy_after) / energy_before
                                  if energy_before > 0 else 0.0
            }

        else:
            # Process in sliding windows
            filtered = torch.zeros_like(values)
            window = self.window_size
            stride = window // 2  # 50% overlap

            total_energy_before = 0.0
            total_energy_after = 0.0
            num_windows = 0

            for start in range(0, S - window + 1, stride):
                end = start + window

                # Extract window
                attn_window = attention_weights[:, :, start:end, start:end]
                values_window = values[:, :, start:end, :]

                # Compute Laplacian for window
                laplacian = self.compute_laplacian(attn_window)

                # Energies
                energy_before = self.compute_dirichlet_energy(values_window, laplacian)

                # Filter window
                if self.method == 'tikhonov':
                    filtered_window = self.tikhonov_filter(values_window, laplacian)
                else:
                    filtered_window = self.spectral_filter(values_window, laplacian)

                energy_after = self.compute_dirichlet_energy(filtered_window, laplacian)

                # Average overlapping regions
                if start == 0:
                    filtered[:, :, start:end, :] = filtered_window
                else:
                    # Blend with previous window in overlap region
                    overlap_start = start
                    overlap_end = start + stride
                    alpha = torch.linspace(0, 1, stride, device=values.device)
                    alpha = alpha.view(1, 1, -1, 1)

                    filtered[:, :, overlap_start:overlap_end, :] = (
                        (1 - alpha) * filtered[:, :, overlap_start:overlap_end, :] +
                        alpha * filtered_window[:, :, :stride, :]
                    )
                    filtered[:, :, overlap_end:end, :] = filtered_window[:, :, stride:, :]

                total_energy_before += energy_before
                total_energy_after += energy_after
                num_windows += 1

            stats = {
                'energy_before': total_energy_before / num_windows,
                'energy_after': total_energy_after / num_windows,
                'energy_reduction': (total_energy_before - total_energy_after) / num_windows,
                'reduction_ratio': (total_energy_before - total_energy_after) / total_energy_before
                                  if total_energy_before > 0 else 0.0,
                'num_windows': num_windows
            }

        self.energy_reduction.append(stats['energy_reduction'])

        return filtered, stats

    def detect_context_shift(
        self,
        attention_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Detect context shifts using Fiedler vector

        The Fiedler vector (2nd eigenvector of Laplacian) partitions
        the graph into two clusters. Sign changes indicate context shifts.

        Args:
            attention_weights: (batch, heads, seq_len, seq_len)

        Returns:
            (fiedler_vector, shift_score)
        """
        # Compute Laplacian
        laplacian = self.compute_laplacian(attention_weights)

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

        # Fiedler vector is 2nd eigenvector (index 1)
        # eigenvalues are sorted ascending, so λ_0=0 is first
        fiedler = eigenvectors[:, :, :, 1]  # (batch, heads, seq_len)

        # Detect sign changes along sequence
        # Number of sign flips indicates number of context shifts
        sign_changes = (fiedler[:, :, 1:] * fiedler[:, :, :-1]) < 0
        shift_score = sign_changes.float().mean().item()

        return fiedler, shift_score

    def get_statistics(self) -> Dict:
        """Get filtering statistics"""
        if not self.energy_reduction:
            return {}

        return {
            'mean_energy_reduction': float(np.mean(self.energy_reduction)),
            'total_energy_removed': float(np.sum(self.energy_reduction)),
            'num_filtered': len(self.energy_reduction),
            'gamma': self.gamma,
            'method': self.method
        }
```

### Model Wrapper

```python
class HarmonicTransformer:
    """
    Wrapper to apply harmonic filtering to transformer

    Intercepts attention computation and filters values
    """

    def __init__(
        self,
        model,
        layer_indices: List[int],
        gamma: float = 0.5,
        method: str = 'tikhonov'
    ):
        """
        Args:
            model: Base transformer model
            layer_indices: Which layers to apply filtering (e.g., [10, 15, 20])
            gamma: Smoothing strength
            method: Filtering method
        """
        self.model = model
        self.layer_indices = layer_indices
        self.gamma = gamma
        self.method = method

        # Create filters for each layer
        self.filters = {
            idx: HarmonicAttentionFilter(gamma=gamma, method=method)
            for idx in layer_indices
        }

        # Hooks
        self.hooks = []
        self.statistics = []

    def apply_harmonic_filtering(self):
        """Register hooks to apply filtering"""

        for idx in self.layer_indices:
            layer = self.model.model.layers[idx]

            def make_hook(filter_obj):
                def hook_fn(module, input, output):
                    # output is typically (hidden_states, attention_weights, ...)

                    if isinstance(output, tuple) and len(output) >= 2:
                        hidden_states = output[0]
                        attention_weights = output[1]

                        if attention_weights is not None:
                            # Apply harmonic filtering
                            # Note: This is simplified - actual implementation
                            # needs to intercept V before attention

weighted sum
                            # For demonstration, we filter the output
                            filtered, stats = filter_obj.filter_values(
                                attention_weights,
                                hidden_states.unsqueeze(1)  # Add head dim
                            )

                            self.statistics.append(stats)

                            # Replace output
                            return (filtered.squeeze(1),) + output[1:]

                    return output

                return hook_fn

            hook = layer.register_forward_hook(make_hook(self.filters[idx]))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_statistics(self) -> Dict:
        """Aggregate statistics from all filters"""
        all_stats = {}

        for idx, filter_obj in self.filters.items():
            stats = filter_obj.get_statistics()
            all_stats[f'layer_{idx}'] = stats

        return all_stats
```

## Expected Results

### Hypothesis 1: Energy Reduction

**Claim**: Harmonic filtering reduces Dirichlet energy, indicating smoother (more coherent) outputs.

**Metric**: Energy reduction ratio

- Baseline: E_after / E_before ≈ 1.0 (no change)
- Harmonic (γ=0.5): E_after / E_before ≈ 0.6-0.7
- Harmonic (γ=2.0): E_after / E_before ≈ 0.3-0.4

### Hypothesis 2: Hallucination Reduction

**Claim**: High-frequency noise corresponds to hallucinations. Filtering removes them.

**Metric**: Factual accuracy on QA

- Baseline: 75% accuracy
- Harmonic: 85-90% accuracy

### Hypothesis 3: Fiedler Vector Detects Shifts

**Claim**: Sign changes in Fiedler vector correspond to topic/context shifts.

**Metric**: Correlation with human-annotated context boundaries

- Expected: R > 0.7 correlation

### Hypothesis 4: Optimal γ is Task-Dependent

**Claim**: Different tasks have different optimal smoothing strengths.

**Metric**: Performance vs γ

- Creative writing: γ_opt ≈ 0.1-0.3 (preserve diversity)
- Factual QA: γ_opt ≈ 0.5-1.0 (enforce consistency)
- Code: γ_opt ≈ 0.7-1.5 (high structure)

---

**This is your 6th major contribution - combining graph theory + PDEs!** 📊🌊
