#!/usr/bin/env python3
"""
Koopman Operator & Dynamic Mode Decomposition for Transformer Layer Jumping

This module implements DMD-based prediction of transformer hidden states,
treating the layer-by-layer evolution as a dynamical system.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class DMDResult:
    """Results from DMD analysis"""
    eigenvalues: torch.Tensor
    modes: torch.Tensor
    coefficients: torch.Tensor
    rank: int
    reconstruction_error: float


class KoopmanDMD:
    """
    Dynamic Mode Decomposition for transformer layer prediction

    Uses spectral methods to find dominant modes of layer-to-layer evolution,
    enabling analytical prediction of future layer states.
    """

    def __init__(self, rank: Optional[int] = None, truncate_threshold: float = 1e-10):
        """
        Args:
            rank: Number of modes to keep (None = auto-determine)
            truncate_threshold: Threshold for singular value truncation
        """
        self.rank = rank
        self.truncate_threshold = truncate_threshold

        # DMD components
        self.Phi = None  # DMD modes (high-dimensional)
        self.Lambda = None  # Eigenvalues (dynamics)
        self.b = None  # Initial coefficients
        self.omega = None  # Continuous-time eigenvalues

        # Metadata
        self.current_layer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                  'mps' if torch.backends.mps.is_available() else 'cpu')

    def fit(self, trajectory: torch.Tensor, dt: float = 1.0) -> 'KoopmanDMD':
        """
        Fit DMD model to observed trajectory

        Args:
            trajectory: Tensor of shape (num_layers, hidden_dim)
                       e.g., layers 1-15 with dimension 4096
            dt: Time step between snapshots (default: 1 layer)

        Returns:
            self (for chaining)
        """
        trajectory = trajectory.to(self.device)

        if len(trajectory.shape) != 2:
            raise ValueError(f"Expected 2D trajectory, got shape {trajectory.shape}")

        num_snapshots, hidden_dim = trajectory.shape
        self.current_layer = num_snapshots

        # Prepare data matrices
        # X = [x₀, x₁, ..., x_{m-2}]
        # Y = [x₁, x₂, ..., x_{m-1}]
        X = trajectory[:-1, :].T  # (d, m-1)
        Y = trajectory[1:, :].T   # (d, m-1)

        # SVD of X = U Σ V*
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)

        # Determine rank (auto or manual)
        if self.rank is None:
            # Auto-determine rank from singular value decay
            # Keep modes that capture 99% of variance
            cumsum = torch.cumsum(S ** 2, dim=0)
            total_var = cumsum[-1]
            self.rank = torch.sum(cumsum / total_var < 0.99).item() + 1
            self.rank = max(self.rank, 1)  # At least 1 mode

        r = min(self.rank, len(S))

        # Truncate to rank r
        U_r = U[:, :r]
        S_r = S[:r]
        V_r = Vh[:r, :].conj().T

        # Compute reduced Koopman operator: Ã = U_r* Y V_r Σ_r^{-1}
        S_r_inv = torch.diag(1.0 / S_r)
        A_tilde = U_r.T @ Y @ V_r @ S_r_inv

        # Eigendecomposition of Ã: Ã W = W Λ
        eigenvalues, eigenvectors = torch.linalg.eig(A_tilde)

        # DMD modes: Φ = Y V_r Σ_r^{-1} W
        # These are the exact DMD modes in high-dimensional space
        self.Phi = Y @ V_r @ S_r_inv @ eigenvectors
        self.Lambda = eigenvalues

        # Continuous-time eigenvalues: ω = log(λ) / dt
        self.omega = torch.log(self.Lambda) / dt

        # Compute initial coefficients: b = Φ^† x₀
        # Use last observed state as initial condition
        x_init = trajectory[-1, :].unsqueeze(1).to(self.Phi.dtype)
        self.b = torch.linalg.lstsq(self.Phi, x_init).solution

        # Compute reconstruction error
        x_reconstructed = self.reconstruct(trajectory.shape[0] - 1)
        self.reconstruction_error = (
            torch.norm(x_reconstructed - trajectory[-1, :]) /
            torch.norm(trajectory[-1, :])
        ).item()

        return self

    def predict(self, target_layer: int, current_layer: Optional[int] = None) -> torch.Tensor:
        """
        Predict hidden state at target_layer using Koopman evolution

        Args:
            target_layer: Layer index to predict
            current_layer: Current layer (default: last observed)

        Returns:
            Predicted hidden state tensor of shape (hidden_dim,)
        """
        if self.Phi is None:
            raise RuntimeError("Must call fit() before predict()")

        if current_layer is None:
            current_layer = self.current_layer

        delta_t = target_layer - current_layer

        if delta_t < 0:
            raise ValueError(f"Cannot predict backwards: {target_layer} < {current_layer}")

        # Time evolution operator: Λ^Δt
        time_evolution = torch.diag(self.Lambda ** delta_t)

        # Prediction: x(t) = Φ Λ^t b
        prediction = self.Phi @ time_evolution @ self.b

        # Take real part (imaginary parts should cancel in stable systems)
        # Small imaginary components may remain due to numerical errors
        prediction_real = prediction.real.squeeze()

        # Check if imaginary part is significant
        imag_magnitude = torch.abs(prediction.imag).max().item()
        if imag_magnitude > 1e-3:
            print(f"Warning: Significant imaginary component in prediction: {imag_magnitude:.6f}")

        return prediction_real

    def reconstruct(self, time_idx: int) -> torch.Tensor:
        """Reconstruct state at given time index using DMD modes"""
        return self.predict(time_idx + 1, current_layer=1)

    def analyze_spectrum(self) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        Analyze eigenvalue spectrum for interpretability

        Returns:
            - Dictionary of spectrum statistics
            - Eigenvalue magnitudes
            - Eigenvalue phases
        """
        if self.Lambda is None:
            raise RuntimeError("Must call fit() before analyze_spectrum()")

        magnitudes = torch.abs(self.Lambda)
        phases = torch.angle(self.Lambda)

        # Categorize modes
        stable_mask = magnitudes < 1.0
        unstable_mask = magnitudes > 1.0
        persistent_mask = torch.abs(magnitudes - 1.0) < 0.1

        # Growth/decay rates
        growth_rates = torch.log(magnitudes).real

        analysis = {
            'num_modes': len(self.Lambda),
            'stable_modes': stable_mask.sum().item(),
            'unstable_modes': unstable_mask.sum().item(),
            'persistent_modes': persistent_mask.sum().item(),
            'max_magnitude': magnitudes.max().item(),
            'min_magnitude': magnitudes.min().item(),
            'mean_magnitude': magnitudes.mean().item(),
            'mean_phase': phases.mean().item(),
            'max_growth_rate': growth_rates.max().item(),
            'min_growth_rate': growth_rates.min().item(),
        }

        return analysis, magnitudes, phases

    def mode_participation(self) -> torch.Tensor:
        """
        Compute participation of each mode in the dynamics

        Returns amplitude of each mode's contribution
        """
        if self.b is None:
            raise RuntimeError("Must call fit() before mode_participation()")

        # Participation = |b_i| * ||φ_i||
        mode_norms = torch.norm(self.Phi, dim=0)
        coeff_magnitudes = torch.abs(self.b.squeeze())

        participation = mode_norms * coeff_magnitudes
        return participation

    def save(self, path: str):
        """Save DMD model to disk"""
        torch.save({
            'Phi': self.Phi,
            'Lambda': self.Lambda,
            'b': self.b,
            'omega': self.omega,
            'rank': self.rank,
            'current_layer': self.current_layer,
            'reconstruction_error': self.reconstruction_error,
        }, path)

    def load(self, path: str):
        """Load DMD model from disk"""
        checkpoint = torch.load(path)
        self.Phi = checkpoint['Phi']
        self.Lambda = checkpoint['Lambda']
        self.b = checkpoint['b']
        self.omega = checkpoint['omega']
        self.rank = checkpoint['rank']
        self.current_layer = checkpoint['current_layer']
        self.reconstruction_error = checkpoint.get('reconstruction_error', 0.0)
        return self


class KoopmanVisualizer:
    """Visualization tools for Koopman/DMD analysis"""

    @staticmethod
    def plot_eigenvalue_spectrum(eigenvalues: torch.Tensor,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot eigenvalues in complex plane and magnitude distribution

        Args:
            eigenvalues: Complex eigenvalues from DMD
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        eigenvalues = eigenvalues.cpu().numpy()
        real = eigenvalues.real
        imag = eigenvalues.imag
        mags = np.abs(eigenvalues)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Complex plane plot
        scatter = ax1.scatter(real, imag, c=mags, cmap='viridis',
                            s=100, alpha=0.7, edgecolors='black', linewidth=1)
        ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax1.axvline(0, color='k', linestyle='--', alpha=0.3)

        # Unit circle (stability boundary)
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.5,
                linewidth=2, label='|λ|=1 (stability)')

        ax1.set_xlabel('Re(λ)', fontsize=12)
        ax1.set_ylabel('Im(λ)', fontsize=12)
        ax1.set_title('Koopman Eigenvalues in Complex Plane', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('|λ|', fontsize=10)

        # Magnitude distribution
        sorted_mags = np.sort(mags)[::-1]
        colors = ['red' if m > 1.0 else 'blue' if m < 1.0 else 'green'
                 for m in sorted_mags]

        ax2.bar(range(len(sorted_mags)), sorted_mags, color=colors, alpha=0.7)
        ax2.axhline(1.0, color='r', linestyle='--', alpha=0.7,
                   linewidth=2, label='|λ|=1')
        ax2.set_xlabel('Mode Index (sorted)', fontsize=12)
        ax2.set_ylabel('|λ|', fontsize=12)
        ax2.set_title('Eigenvalue Magnitudes', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_mode_participation(participation: torch.Tensor,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot participation of each DMD mode"""
        participation = participation.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Normalize participation to percentages
        participation_pct = 100 * participation / participation.sum()

        ax.bar(range(len(participation_pct)), participation_pct, alpha=0.7)
        ax.set_xlabel('Mode Index', fontsize=12)
        ax.set_ylabel('Participation (%)', fontsize=12)
        ax.set_title('DMD Mode Participation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add cumulative line
        cumsum = np.cumsum(participation_pct)
        ax2 = ax.twinx()
        ax2.plot(range(len(cumsum)), cumsum, 'r-', linewidth=2,
                label='Cumulative')
        ax2.axhline(95, color='r', linestyle='--', alpha=0.5, label='95%')
        ax2.set_ylabel('Cumulative Participation (%)', fontsize=12, color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='lower right')
        ax2.set_ylim(0, 105)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_prediction_accuracy(
        observe_layers_range: List[int],
        target_layers_range: List[int],
        accuracy_matrix: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of prediction accuracy

        Args:
            observe_layers_range: List of observation window sizes
            target_layers_range: List of target layers
            accuracy_matrix: Matrix of accuracies (cosine similarities)
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto',
                      vmin=0.5, vmax=1.0, interpolation='nearest')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', fontsize=12)

        # Labels
        ax.set_xlabel('Target Layer', fontsize=12)
        ax.set_ylabel('Observed Layers', fontsize=12)
        ax.set_title('DMD Prediction Accuracy Heatmap', fontsize=14, fontweight='bold')

        # Ticks
        ax.set_xticks(range(len(target_layers_range)))
        ax.set_xticklabels(target_layers_range)
        ax.set_yticks(range(len(observe_layers_range)))
        ax.set_yticklabels(observe_layers_range)

        # Annotate cells
        for i in range(len(observe_layers_range)):
            for j in range(len(target_layers_range)):
                if accuracy_matrix[i, j] > 0:
                    text_color = 'white' if accuracy_matrix[i, j] < 0.75 else 'black'
                    ax.text(j, i, f'{accuracy_matrix[i, j]:.2f}',
                           ha='center', va='center', color=text_color, fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


def demo_dmd_on_synthetic_data():
    """
    Demonstrate DMD on synthetic dynamical system
    Useful for validation before applying to real transformer data
    """
    print("="*60)
    print("DMD Demo on Synthetic Dynamical System")
    print("="*60)

    # Create synthetic linear system with known eigenvalues
    torch.manual_seed(42)
    dim = 100
    num_snapshots = 20

    # True eigenvalues (design the dynamics)
    true_lambdas = torch.tensor([
        0.95 + 0.1j,  # Stable spiral
        0.95 - 0.1j,  # Stable spiral (conjugate)
        0.98,          # Slow decay
        0.85,          # Fast decay
        1.02,          # Slight growth
    ], dtype=torch.complex64)

    # Random modes
    true_modes = torch.randn(dim, len(true_lambdas), dtype=torch.complex64)
    true_modes, _ = torch.linalg.qr(true_modes)  # Orthonormalize

    # Initial coefficients
    true_b = torch.randn(len(true_lambdas), 1, dtype=torch.complex64)

    # Generate trajectory
    trajectory = []
    for t in range(num_snapshots):
        state = true_modes @ torch.diag(true_lambdas ** t) @ true_b
        trajectory.append(state.real.squeeze())

    trajectory = torch.stack(trajectory)

    # Fit DMD
    print(f"\nTrajectory shape: {trajectory.shape}")
    print(f"True eigenvalues: {true_lambdas}")

    dmd = KoopmanDMD(rank=5)
    dmd.fit(trajectory[:15])  # Use first 15 snapshots

    print(f"\nRecovered eigenvalues: {dmd.Lambda}")
    print(f"Reconstruction error: {dmd.reconstruction_error:.6f}")

    # Test prediction
    predicted = dmd.predict(target_layer=19, current_layer=15)
    true_final = trajectory[19]

    cosine_sim = F.cosine_similarity(
        predicted.unsqueeze(0),
        true_final.unsqueeze(0)
    ).item()

    rel_error = (torch.norm(predicted - true_final) / torch.norm(true_final)).item()

    print(f"\nPrediction (layer 15 → 19):")
    print(f"  Cosine similarity: {cosine_sim:.6f}")
    print(f"  Relative error: {rel_error:.6f}")

    # Analyze spectrum
    analysis, mags, phases = dmd.analyze_spectrum()
    print(f"\nSpectrum analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    # Visualize
    fig1 = KoopmanVisualizer.plot_eigenvalue_spectrum(dmd.Lambda)
    fig1.suptitle("DMD on Synthetic System", fontsize=16, fontweight='bold')
    plt.show()

    print("\n✅ DMD validation complete!")


if __name__ == "__main__":
    # Run demo
    demo_dmd_on_synthetic_data()
