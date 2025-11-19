#!/usr/bin/env python3
"""
Test Optimal Transport Attention Reweighting

Validates Sinkhorn algorithm and OT-based attention refinement
"""

import torch
import torch.nn.functional as F
from optimal_transport_attention import (
    OptimalTransportAttention,
    OTPresets,
    SinkhornStats
)
import numpy as np
import matplotlib.pyplot as plt
import argparse


class OTTester:
    """Test harness for optimal transport attention"""

    def __init__(self):
        """Initialize tester"""
        print("Optimal Transport Attention Tester")
        print("=" * 60)

    def test_sinkhorn_convergence(self):
        """Test 1: Sinkhorn algorithm convergence"""
        print("\n" + "=" * 60)
        print("TEST 1: Sinkhorn Convergence")
        print("=" * 60)

        seq_len = 32
        device = torch.device("cpu")

        # Create simple cost matrix (distance-based)
        i_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        j_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        C = torch.abs(i_idx - j_idx).float()

        # Uniform marginals
        a = torch.ones(seq_len, device=device) / seq_len
        b = torch.ones(seq_len, device=device) / seq_len

        # Test different lambda values
        lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]

        print(f"\nTesting Sinkhorn with different regularization parameters:")
        print(f"Cost matrix: distance-based ({seq_len}x{seq_len})")
        print(f"Marginals: uniform\n")

        results = []

        for lam in lambdas:
            ot = OptimalTransportAttention(reg_lambda=lam, max_iterations=200)
            P, stats = ot.sinkhorn(C, a, b)

            results.append({
                'lambda': lam,
                'iterations': stats.iterations,
                'wasserstein': stats.wasserstein_distance,
                'entropy': stats.entropy_after,
                'conv_error': stats.convergence_error
            })

            print(f"λ = {lam:5.2f}:")
            print(f"  Iterations: {stats.iterations:3d}")
            print(f"  Wasserstein: {stats.wasserstein_distance:8.4f}")
            print(f"  Entropy: {stats.entropy_after:8.4f}")
            print(f"  Conv error: {stats.convergence_error:8.2e}")

        # Verify: smaller lambda → lower entropy (sharper distribution)
        entropies = [r['entropy'] for r in results]
        entropy_decreasing = all(entropies[i] >= entropies[i+1] for i in range(len(entropies)-1))

        print(f"\nEntropy decreases with lambda: {'✅ PASS' if entropy_decreasing else '❌ FAIL'}")

        return results

    def test_attention_reweighting(self):
        """Test 2: Attention matrix reweighting"""
        print("\n" + "=" * 60)
        print("TEST 2: Attention Reweighting")
        print("=" * 60)

        seq_len = 24

        # Create peaked attention (low entropy)
        attention = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            # Each row attends strongly to just 2-3 tokens
            attention[i, max(0, i-1):min(seq_len, i+2)] = 1.0

        attention = F.softmax(attention * 5.0, dim=1)  # Make it very peaked

        print(f"\nOriginal attention:")
        entropy_orig = -(attention * torch.log(attention + 1e-10)).sum(dim=1).mean()
        print(f"  Shape: {attention.shape}")
        print(f"  Mean row entropy: {entropy_orig:.4f}")
        print(f"  Min value: {attention.min():.6f}")
        print(f"  Max value: {attention.max():.6f}")

        # Apply OT reweighting
        ot = OptimalTransportAttention(
            reg_lambda=0.1,
            cost_type='distance',
            target_distribution='uniform'
        )

        reweighted, stats = ot.reweight_attention(attention)

        print(f"\nReweighted attention:")
        entropy_new = -(reweighted * torch.log(reweighted + 1e-10)).sum(dim=1).mean()
        print(f"  Mean row entropy: {entropy_new:.4f}")
        print(f"  Entropy increase: {entropy_new - entropy_orig:.4f}")
        print(f"  Min value: {reweighted.min():.6f}")
        print(f"  Max value: {reweighted.max():.6f}")

        print(f"\nSinkhorn stats:")
        print(f"  Iterations: {stats.iterations}")
        print(f"  Wasserstein: {stats.wasserstein_distance:.4f}")
        print(f"  Convergence error: {stats.convergence_error:.2e}")

        # Verify: entropy increased (distribution is smoother)
        success = entropy_new > entropy_orig

        print(f"\nEntropy increased: {'✅ PASS' if success else '❌ FAIL'}")

        return success

    def test_preset_comparison(self):
        """Test 3: Compare different presets"""
        print("\n" + "=" * 60)
        print("TEST 3: Preset Comparison")
        print("=" * 60)

        seq_len = 20

        # Create attention with some structure
        attention = torch.randn(seq_len, seq_len)
        attention = F.softmax(attention, dim=1)

        presets = ['light', 'medium', 'heavy', 'locality']
        results = {}

        print(f"\nOriginal attention entropy: {-(attention * torch.log(attention + 1e-10)).sum():.4f}\n")

        for preset_name in presets:
            print(f"{'-'*40}")
            print(f"Preset: {preset_name.upper()}")
            print(f"{'-'*40}")

            config = OTPresets.get_preset(preset_name)
            ot = OptimalTransportAttention(**config)

            reweighted, stats = ot.reweight_attention(attention)

            results[preset_name] = {
                'wasserstein': stats.wasserstein_distance,
                'entropy_before': stats.entropy_before,
                'entropy_after': stats.entropy_after,
                'entropy_increase': stats.entropy_after - stats.entropy_before,
                'iterations': stats.iterations
            }

            print(f"  Wasserstein: {stats.wasserstein_distance:.4f}")
            print(f"  Entropy before: {stats.entropy_before:.4f}")
            print(f"  Entropy after: {stats.entropy_after:.4f}")
            print(f"  Increase: {stats.entropy_after - stats.entropy_before:+.4f}")
            print(f"  Iterations: {stats.iterations}")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        for preset_name, result in results.items():
            print(f"{preset_name.upper():12s}: {result['entropy_increase']:+.4f} entropy change")

        return results

    def test_cost_matrix_types(self):
        """Test 4: Different cost matrix types"""
        print("\n" + "=" * 60)
        print("TEST 4: Cost Matrix Types")
        print("=" * 60)

        seq_len = 16
        attention = torch.rand(seq_len, seq_len)
        attention = F.softmax(attention, dim=1)

        cost_types = ['uniform', 'distance', 'entropy']
        results = {}

        for cost_type in cost_types:
            print(f"\n{'-'*40}")
            print(f"Cost type: {cost_type}")
            print(f"{'-'*40}")

            ot = OptimalTransportAttention(
                reg_lambda=0.1,
                cost_type=cost_type
            )

            # Compute cost matrix
            C = ot.compute_cost_matrix(seq_len, attention)

            print(f"  Cost matrix shape: {C.shape}")
            print(f"  Cost range: [{C.min():.4f}, {C.max():.4f}]")
            print(f"  Cost mean: {C.mean():.4f}")

            # Reweight
            reweighted, stats = ot.reweight_attention(attention)

            results[cost_type] = {
                'cost_min': C.min().item(),
                'cost_max': C.max().item(),
                'wasserstein': stats.wasserstein_distance,
                'entropy_change': stats.entropy_after - stats.entropy_before
            }

            print(f"  Wasserstein: {stats.wasserstein_distance:.4f}")
            print(f"  Entropy change: {stats.entropy_after - stats.entropy_before:+.4f}")

        return results

    def test_marginal_preservation(self):
        """Test 5: Verify marginal constraints are satisfied"""
        print("\n" + "=" * 60)
        print("TEST 5: Marginal Constraint Preservation")
        print("=" * 60)

        seq_len = 24

        # Create non-uniform source distribution
        a = torch.softmax(torch.randn(seq_len), dim=0)

        # Uniform target
        b = torch.ones(seq_len) / seq_len

        # Distance cost
        i_idx = torch.arange(seq_len).unsqueeze(1)
        j_idx = torch.arange(seq_len).unsqueeze(0)
        C = torch.abs(i_idx - j_idx).float()

        ot = OptimalTransportAttention(reg_lambda=0.1)
        P, stats = ot.sinkhorn(C, a, b)

        # Check marginals
        row_sums = P.sum(dim=1)
        col_sums = P.sum(dim=0)

        row_error = torch.max(torch.abs(row_sums - a)).item()
        col_error = torch.max(torch.abs(col_sums - b)).item()

        print(f"\nMarginal constraints:")
        print(f"  Target row marginal: non-uniform")
        print(f"  Target col marginal: uniform")

        print(f"\nActual marginals:")
        print(f"  Row sum error: {row_error:.2e}")
        print(f"  Col sum error: {col_error:.2e}")

        print(f"\nConvergence:")
        print(f"  Iterations: {stats.iterations}")
        print(f"  Convergence error: {stats.convergence_error:.2e}")

        # Success if errors are small
        success = row_error < 1e-4 and col_error < 1e-4

        print(f"\nMarginals preserved: {'✅ PASS' if success else '❌ FAIL'}")

        return success

    def test_entropy_regularization_effect(self):
        """Test 6: Effect of entropy regularization parameter"""
        print("\n" + "=" * 60)
        print("TEST 6: Entropy Regularization Effect")
        print("=" * 60)

        seq_len = 20
        attention = torch.rand(seq_len, seq_len)
        attention = F.softmax(attention * 3.0, dim=1)  # Somewhat peaked

        lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]

        print(f"\nOriginal attention:")
        orig_entropy = -(attention * torch.log(attention + 1e-10)).sum()
        print(f"  Entropy: {orig_entropy:.4f}")

        print(f"\nTesting different λ values:\n")

        results = []

        for lam in lambdas:
            ot = OptimalTransportAttention(reg_lambda=lam, cost_type='distance')
            reweighted, stats = ot.reweight_attention(attention)

            entropy_change = stats.entropy_after - stats.entropy_before

            results.append({
                'lambda': lam,
                'entropy_after': stats.entropy_after,
                'entropy_change': entropy_change,
                'wasserstein': stats.wasserstein_distance
            })

            print(f"λ = {lam:5.2f}:")
            print(f"  Entropy after: {stats.entropy_after:7.4f}")
            print(f"  Entropy change: {entropy_change:+7.4f}")
            print(f"  Wasserstein: {stats.wasserstein_distance:7.4f}")

        # Analysis: larger lambda → larger entropy (more regularization)
        entropies = [r['entropy_after'] for r in results]
        entropy_increasing = all(entropies[i] <= entropies[i+1] for i in range(len(entropies)-1))

        print(f"\nEntropy increases with λ: {'✅ PASS' if entropy_increasing else '❌ FAIL'}")

        return results

    def visualize_reweighting(self):
        """Test 7: Visualize attention reweighting"""
        print("\n" + "=" * 60)
        print("TEST 7: Attention Reweighting Visualization")
        print("=" * 60)

        seq_len = 32

        # Create structured attention (causal with noise)
        attention = torch.tril(torch.ones(seq_len, seq_len))
        attention = attention + 0.1 * torch.rand(seq_len, seq_len)
        attention = F.softmax(attention * 2.0, dim=1)

        # Apply OT
        ot = OptimalTransportAttention(
            reg_lambda=0.1,
            cost_type='distance'
        )

        reweighted, stats = ot.reweight_attention(attention)

        print(f"\nStatistics:")
        print(f"  Wasserstein distance: {stats.wasserstein_distance:.4f}")
        print(f"  Entropy before: {stats.entropy_before:.4f}")
        print(f"  Entropy after: {stats.entropy_after:.4f}")
        print(f"  Increase: {stats.entropy_after - stats.entropy_before:+.4f}")

        # Visualize
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original
            im1 = axes[0].imshow(attention.numpy(), cmap='viridis', aspect='auto')
            axes[0].set_title('Original Attention')
            axes[0].set_xlabel('Key')
            axes[0].set_ylabel('Query')
            plt.colorbar(im1, ax=axes[0])

            # Reweighted
            im2 = axes[1].imshow(reweighted.numpy(), cmap='viridis', aspect='auto')
            axes[1].set_title('OT Reweighted')
            axes[1].set_xlabel('Key')
            axes[1].set_ylabel('Query')
            plt.colorbar(im2, ax=axes[1])

            # Difference
            diff = reweighted - attention
            im3 = axes[2].imshow(diff.numpy(), cmap='RdBu', aspect='auto',
                                vmin=-diff.abs().max(), vmax=diff.abs().max())
            axes[2].set_title('Difference (Reweighted - Original)')
            axes[2].set_xlabel('Key')
            axes[2].set_ylabel('Query')
            plt.colorbar(im3, ax=axes[2])

            plt.tight_layout()
            plt.savefig('ot_attention_reweighting.png', dpi=150, bbox_inches='tight')
            print(f"\n  Visualization saved to: ot_attention_reweighting.png")
            plt.close()

        except Exception as e:
            print(f"  Plotting failed: {e}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Test optimal transport attention")
    parser.add_argument("--test", type=str, default="all",
                       choices=["convergence", "reweight", "presets", "cost",
                               "marginals", "lambda", "viz", "all"],
                       help="Which test to run")

    args = parser.parse_args()

    tester = OTTester()

    if args.test in ["convergence", "all"]:
        tester.test_sinkhorn_convergence()

    if args.test in ["reweight", "all"]:
        tester.test_attention_reweighting()

    if args.test in ["presets", "all"]:
        tester.test_preset_comparison()

    if args.test in ["cost", "all"]:
        tester.test_cost_matrix_types()

    if args.test in ["marginals", "all"]:
        tester.test_marginal_preservation()

    if args.test in ["lambda", "all"]:
        tester.test_entropy_regularization_effect()

    if args.test in ["viz", "all"]:
        tester.visualize_reweighting()

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
