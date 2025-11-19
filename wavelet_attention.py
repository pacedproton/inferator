#!/usr/bin/env python3
"""
Wavelet Transform for Hierarchical Attention

Uses multi-resolution wavelet analysis to capture structure at multiple scales.

Mathematical foundation:
- Wavelet theory (Daubechies wavelets)
- Multi-resolution analysis (MRA)
- Fast Wavelet Transform (FWT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt  # PyWavelets
from typing import List, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass


@dataclass
class WaveletStats:
    """Statistics from wavelet decomposition"""
    num_levels: int
    wavelet_type: str
    detail_energies: List[float]
    approximation_energy: float
    total_energy: float
    compression_ratio: float


class WaveletAttention:
    """
    Multi-scale attention using wavelet decomposition

    Key idea: Decompose sequences into multiple scales (coarse + details),
    apply attention at each scale, then reconstruct.

    Wavelet decomposition:
    x = a_J φ_J + Σ_{j=1}^J d_j ψ_j

    where:
    - a_J = approximation coefficients (low freq / coarse scale)
    - d_j = detail coefficients (high freq / fine scale)
    - φ_J = scaling function
    - ψ_j = wavelet functions

    Complexity: O(n log n) with Fast Wavelet Transform
    """

    def __init__(
        self,
        wavelet: str = 'db4',
        max_level: Optional[int] = None,
        mode: str = 'periodization',
        threshold_type: str = 'soft',
        threshold_scale: float = 0.0
    ):
        """
        Args:
            wavelet: Wavelet type ('db4', 'haar', 'sym4', etc.)
            max_level: Maximum decomposition level (None = auto)
            mode: Padding mode for DWT
            threshold_type: 'soft', 'hard', or None
            threshold_scale: Threshold for denoising (0 = no thresholding)
        """
        self.wavelet = wavelet
        self.max_level = max_level
        self.mode = mode
        self.threshold_type = threshold_type
        self.threshold_scale = threshold_scale

        # Verify wavelet exists
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Unknown wavelet: {wavelet}. Available: {pywt.wavelist()}")

        print(f"WaveletAttention initialized:")
        print(f"  Wavelet: {wavelet}")
        print(f"  Max level: {max_level if max_level else 'auto'}")
        print(f"  Mode: {mode}")
        if threshold_scale > 0:
            print(f"  Threshold: {threshold_type} @ {threshold_scale}")

    def decompose_1d(
        self,
        signal: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, int]:
        """
        1D wavelet decomposition

        Args:
            signal: 1D tensor of shape (seq_len,)

        Returns:
            (detail_coeffs, approx_coeffs, num_levels)
            detail_coeffs: List of detail coefficients [d₁, d₂, ..., d_J]
            approx_coeffs: Approximation coefficients a_J
        """
        # Convert to numpy for pywt
        signal_np = signal.detach().cpu().numpy()

        # Determine max level if not specified
        max_level = self.max_level
        if max_level is None:
            max_level = pywt.dwt_max_level(len(signal_np), self.wavelet)

        # Multilevel DWT
        coeffs = pywt.wavedec(signal_np, self.wavelet, mode=self.mode, level=max_level)

        # coeffs = [a_J, d_J, d_{J-1}, ..., d_1]
        approx = torch.from_numpy(coeffs[0]).to(signal.device)
        details = [torch.from_numpy(c).to(signal.device) for c in coeffs[1:]]

        return details, approx, max_level

    def reconstruct_1d(
        self,
        details: List[torch.Tensor],
        approx: torch.Tensor
    ) -> torch.Tensor:
        """
        1D wavelet reconstruction

        Args:
            details: List of detail coefficients
            approx: Approximation coefficients

        Returns:
            Reconstructed signal
        """
        # Convert to numpy
        approx_np = approx.detach().cpu().numpy()
        details_np = [d.detach().cpu().numpy() for d in details]

        # Reconstruct
        coeffs = [approx_np] + details_np
        signal_np = pywt.waverec(coeffs, self.wavelet, mode=self.mode)

        return torch.from_numpy(signal_np).to(approx.device)

    def threshold_coefficients(
        self,
        coeffs: List[torch.Tensor],
        threshold: float,
        threshold_type: str = 'soft'
    ) -> List[torch.Tensor]:
        """
        Apply thresholding to wavelet coefficients for denoising

        Args:
            coeffs: List of coefficient tensors
            threshold: Threshold value
            threshold_type: 'soft' or 'hard'

        Returns:
            Thresholded coefficients
        """
        if threshold_type == 'soft':
            # Soft thresholding: sign(x) * max(|x| - threshold, 0)
            thresholded = [
                torch.sign(c) * torch.maximum(torch.abs(c) - threshold, torch.zeros_like(c))
                for c in coeffs
            ]
        elif threshold_type == 'hard':
            # Hard thresholding: x if |x| > threshold else 0
            thresholded = [
                c * (torch.abs(c) > threshold).float()
                for c in coeffs
            ]
        else:
            thresholded = coeffs

        return thresholded

    def compute_energy(
        self,
        details: List[torch.Tensor],
        approx: torch.Tensor
    ) -> WaveletStats:
        """
        Compute energy distribution across scales

        Args:
            details: Detail coefficients
            approx: Approximation coefficients

        Returns:
            Statistics including energy per scale
        """
        detail_energies = [torch.sum(d**2).item() for d in details]
        approx_energy = torch.sum(approx**2).item()
        total_energy = approx_energy + sum(detail_energies)

        # Compression ratio (if thresholding)
        if self.threshold_scale > 0:
            num_nonzero = sum((torch.abs(d) > self.threshold_scale).sum().item() for d in details)
            num_nonzero += (torch.abs(approx) > self.threshold_scale).sum().item()
            total_coeffs = sum(d.numel() for d in details) + approx.numel()
            compression_ratio = total_coeffs / (num_nonzero + 1)
        else:
            compression_ratio = 1.0

        return WaveletStats(
            num_levels=len(details),
            wavelet_type=self.wavelet,
            detail_energies=detail_energies,
            approximation_energy=approx_energy,
            total_energy=total_energy,
            compression_ratio=compression_ratio
        )

    def multiscale_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Multi-scale attention using wavelet decomposition

        Args:
            query: (seq_len, d_model)
            key: (seq_len, d_model)
            value: (seq_len, d_model)

        Returns:
            (output, stats_dict)
        """
        seq_len, d_model = query.shape
        device = query.device

        # Process each dimension independently (simplified)
        # In practice, you might want more sophisticated multi-dimensional wavelets

        outputs = []
        all_stats = []

        for dim in range(d_model):
            # Decompose Q, K, V for this dimension
            q_details, q_approx, num_levels = self.decompose_1d(query[:, dim])
            k_details, k_approx, _ = self.decompose_1d(key[:, dim])
            v_details, v_approx, _ = self.decompose_1d(value[:, dim])

            # Attention at approximation level (coarse scale)
            attn_approx = self._compute_attention_1d(q_approx, k_approx, v_approx)

            # Attention at each detail level (fine scales)
            attn_details = []
            for q_d, k_d, v_d in zip(q_details, k_details, v_details):
                attn_d = self._compute_attention_1d(q_d, k_d, v_d)
                attn_details.append(attn_d)

            # Apply thresholding if specified
            if self.threshold_scale > 0:
                attn_details = self.threshold_coefficients(
                    attn_details, self.threshold_scale, self.threshold_type
                )

            # Reconstruct
            output_dim = self.reconstruct_1d(attn_details, attn_approx)

            # Truncate/pad to original length
            if len(output_dim) > seq_len:
                output_dim = output_dim[:seq_len]
            elif len(output_dim) < seq_len:
                output_dim = F.pad(output_dim, (0, seq_len - len(output_dim)))

            outputs.append(output_dim)

            # Compute stats
            stats = self.compute_energy(attn_details, attn_approx)
            all_stats.append(stats)

        # Stack outputs
        output = torch.stack(outputs, dim=1)

        # Aggregate stats
        agg_stats = {
            'num_levels': all_stats[0].num_levels,
            'mean_approx_energy': np.mean([s.approximation_energy for s in all_stats]),
            'mean_detail_energies': [
                np.mean([s.detail_energies[i] for s in all_stats])
                for i in range(all_stats[0].num_levels)
            ],
            'mean_compression': np.mean([s.compression_ratio for s in all_stats])
        }

        return output, agg_stats

    def _compute_attention_1d(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention for 1D signals at single scale

        Args:
            q, k, v: 1D tensors (possibly different lengths)

        Returns:
            Attention output (same length as q)
        """
        len_q = len(q)
        len_k = len(k)

        if len_q == 0 or len_k == 0:
            return torch.zeros_like(q)

        # Compute scores: outer product
        scores = torch.outer(q, k)  # (len_q, len_k)

        # Softmax
        attn_weights = F.softmax(scores, dim=1)

        # Weighted sum of values
        output = attn_weights @ v  # (len_q,)

        return output


class WaveletMultiHeadAttention(nn.Module):
    """
    Multi-head attention with wavelet decomposition at each scale

    Combines standard multi-head attention with wavelet MRA
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        wavelet_config: dict,
        use_wavelets: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            wavelet_config: Configuration for WaveletAttention
            use_wavelets: If False, use standard attention
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_wavelets = use_wavelets

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Wavelet processor
        if use_wavelets:
            self.wavelet = WaveletAttention(**wavelet_config)

        print(f"WaveletMultiHeadAttention:")
        print(f"  Model dim: {d_model}")
        print(f"  Num heads: {num_heads}")
        print(f"  Use wavelets: {use_wavelets}")

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass

        Args:
            x: (batch, seq_len, d_model)
            return_stats: Whether to return wavelet statistics

        Returns:
            (output, stats_dict)
        """
        batch, seq_len, _ = x.shape

        # Project
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to multi-head
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_wavelets:
            # Apply wavelet attention per head
            outputs = []
            all_stats = []

            for b in range(batch):
                for h in range(self.num_heads):
                    q_h = Q[b, h, :, :]  # (seq_len, head_dim)
                    k_h = K[b, h, :, :]
                    v_h = V[b, h, :, :]

                    out_h, stats = self.wavelet.multiscale_attention(q_h, k_h, v_h)
                    outputs.append(out_h)
                    all_stats.append(stats)

            # Reshape
            output = torch.stack(outputs).view(batch, self.num_heads, seq_len, self.head_dim)

            # Aggregate stats
            if return_stats:
                stats_dict = {
                    'num_levels': all_stats[0]['num_levels'],
                    'mean_compression': np.mean([s['mean_compression'] for s in all_stats])
                }
            else:
                stats_dict = None

        else:
            # Standard attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V)
            stats_dict = None

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.out_proj(output)

        return output, stats_dict


# Preset configurations
class WaveletPresets:
    """Recommended wavelet settings"""

    # Haar wavelet: simplest, fastest
    HAAR = {
        'wavelet': 'haar',
        'max_level': None,
        'mode': 'periodization',
        'threshold_type': None,
        'threshold_scale': 0.0
    }

    # Daubechies 4: good balance
    DB4 = {
        'wavelet': 'db4',
        'max_level': None,
        'mode': 'periodization',
        'threshold_type': None,
        'threshold_scale': 0.0
    }

    # Symlet 8: symmetric, smooth
    SYM8 = {
        'wavelet': 'sym8',
        'max_level': None,
        'mode': 'periodization',
        'threshold_type': None,
        'threshold_scale': 0.0
    }

    # Denoising: soft thresholding
    DENOISE = {
        'wavelet': 'db4',
        'max_level': 3,
        'mode': 'periodization',
        'threshold_type': 'soft',
        'threshold_scale': 0.1
    }

    @classmethod
    def get_preset(cls, name: str) -> dict:
        """Get preset configuration"""
        presets = {
            'haar': cls.HAAR,
            'db4': cls.DB4,
            'sym8': cls.SYM8,
            'denoise': cls.DENOISE
        }

        if name.lower() not in presets:
            raise ValueError(f"Unknown preset: {name}. Choose from: {list(presets.keys())}")

        return presets[name.lower()]


if __name__ == "__main__":
    print("Wavelet Transform for Hierarchical Attention")
    print("=" * 60)
    print("\nMulti-resolution analysis for capturing structure at multiple scales\n")

    # Demo
    print("Demo: Wavelet decomposition of 1D signal\n")

    seq_len = 128
    signal = torch.randn(seq_len)

    wavelet = WaveletAttention(wavelet='db4', max_level=3)

    details, approx, num_levels = wavelet.decompose_1d(signal)

    print(f"Signal length: {seq_len}")
    print(f"Num levels: {num_levels}")
    print(f"Approximation length: {len(approx)}")
    print(f"Detail lengths: {[len(d) for d in details]}")

    # Compute energy
    stats = wavelet.compute_energy(details, approx)

    print(f"\nEnergy distribution:")
    print(f"  Approximation: {stats.approximation_energy:.4f}")
    for i, e in enumerate(stats.detail_energies):
        print(f"  Detail level {i+1}: {e:.4f}")
    print(f"  Total: {stats.total_energy:.4f}")

    # Reconstruct
    reconstructed = wavelet.reconstruct_1d(details, approx)
    error = torch.norm(signal - reconstructed[:seq_len]) / torch.norm(signal)

    print(f"\nReconstruction error: {error:.2e}")

    print("\nSee test_wavelet.py for full testing")
