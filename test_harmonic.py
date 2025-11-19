#!/usr/bin/env python3
"""
Test Harmonic Attention Filtering

Demonstrates spectral denoising of attention values
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import numpy as np
from harmonic_attention_filter import (
    HarmonicAttentionFilter,
    HarmonicTransformer,
    FilterPresets
)
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class HarmonicTester:
    """Test harness for harmonic filtering"""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        """Load model and tokenizer"""
        print(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-bit quantization for 64GB Mac
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            output_attentions=True  # Critical for harmonic filtering
        )

        self.model.eval()
        print("Model loaded successfully\n")

    def extract_attention_and_values(
        self,
        prompt: str,
        layer_idx: int = 15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention weights and values from a specific layer

        Args:
            prompt: Input text
            layer_idx: Which layer to extract from

        Returns:
            (attention_weights, values) both as tensors
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Hook to capture values
        values_captured = {}

        def capture_values(module, input, output):
            # For HuggingFace self-attention modules
            # input[0] is the hidden states
            # We need to capture the values before projection
            hidden_states = input[0]
            values_captured['values'] = hidden_states
            return output

        # Register hook
        layer = self.model.model.layers[layer_idx].self_attn
        hook = layer.register_forward_hook(capture_values)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Extract attention weights
        attention_weights = outputs.attentions[layer_idx]  # (batch, heads, seq, seq)

        # Get values (approximate from hidden states)
        values = values_captured.get('values', None)

        # Remove hook
        hook.remove()

        return attention_weights, values

    def test_basic_filtering(
        self,
        gamma: float = 0.1,
        method: str = "tikhonov"
    ):
        """Test 1: Basic filtering on a simple prompt"""
        print("="*60)
        print("TEST 1: Basic Harmonic Filtering")
        print("="*60)

        prompt = "The Eiffel Tower is located in Paris, France. It was built in"

        # Extract attention and values
        print(f"\nPrompt: '{prompt}'")
        print("Extracting attention weights...")

        attention_weights, _ = self.extract_attention_and_values(prompt, layer_idx=15)

        # Use first head for simplicity
        A = attention_weights[0, 0, :, :].cpu()  # (seq_len, seq_len)
        seq_len = A.shape[0]

        print(f"  Attention shape: {A.shape}")
        print(f"  Sequence length: {seq_len}")

        # Create filter
        filter = HarmonicAttentionFilter(gamma=gamma, method=method)

        # Compute Laplacian
        print("\nComputing graph Laplacian...")
        L = filter.compute_laplacian(A, normalize=True)

        print(f"  Laplacian shape: {L.shape}")
        print(f"  Laplacian norm: {L.norm():.4f}")

        # Create dummy values (random for demonstration)
        V = torch.randn(seq_len, 64)  # (seq_len, head_dim)

        # Measure energy before
        energy_before = filter.compute_dirichlet_energy(V, L)
        print(f"\nDirichlet energy BEFORE: {energy_before:.6f}")

        # Apply filter
        print(f"\nApplying {method} filter (gamma={gamma})...")
        if method == "tikhonov":
            V_filtered = filter.tikhonov_filter(V, L, gamma)
        else:
            V_filtered, num_high_freq, spectral_gap = filter.spectral_filter(V, L, 0.3)
            print(f"  High frequencies zeroed: {num_high_freq}")
            print(f"  Spectral gap: {spectral_gap:.6f}")

        # Measure energy after
        energy_after = filter.compute_dirichlet_energy(V_filtered, L)
        print(f"\nDirichlet energy AFTER: {energy_after:.6f}")

        reduction = 1.0 - (energy_after / energy_before)
        print(f"Energy reduction: {reduction*100:.2f}%")

        # Success if energy reduced significantly
        success = reduction > 0.1
        print(f"\nFiltering: {'✅ SUCCESS' if success else '❌ FAILED'}")

        return success

    def test_spectral_analysis(self):
        """Test 2: Analyze eigenvalue spectrum of attention Laplacian"""
        print("\n" + "="*60)
        print("TEST 2: Spectral Analysis")
        print("="*60)

        prompt = "In machine learning, transformers are models that use attention mechanisms to"

        # Extract attention
        print(f"\nPrompt: '{prompt}'")
        attention_weights, _ = self.extract_attention_and_values(prompt, layer_idx=15)

        A = attention_weights[0, 0, :, :].cpu()

        # Create filter
        filter = HarmonicAttentionFilter(gamma=0.1, method="spectral")

        # Compute Laplacian
        L = filter.compute_laplacian(A, normalize=True)

        # Eigendecomposition
        print("\nComputing eigendecomposition...")
        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        eigenvalues = eigenvalues.numpy()

        print(f"\nEigenvalue statistics:")
        print(f"  Min: {eigenvalues[0]:.6f}")
        print(f"  Max: {eigenvalues[-1]:.6f}")
        print(f"  Mean: {eigenvalues.mean():.6f}")
        print(f"  Median: {np.median(eigenvalues):.6f}")

        # Spectral gap (important for graph connectivity)
        spectral_gap = eigenvalues[1] - eigenvalues[0]
        print(f"  Spectral gap: {spectral_gap:.6f}")

        # Fiedler vector (2nd eigenvector)
        fiedler = eigenvectors[:, 1].numpy()
        print(f"\nFiedler vector:")
        print(f"  Shape: {fiedler.shape}")
        print(f"  Sign changes: {np.sum(np.diff(np.sign(fiedler)) != 0)}")

        # Visualize if possible
        try:
            self._plot_spectrum(eigenvalues, fiedler)
        except Exception as e:
            print(f"Plotting failed: {e}")

        return True

    def test_context_shift_detection(self):
        """Test 3: Detect topic boundaries using Fiedler vector"""
        print("\n" + "="*60)
        print("TEST 3: Context Shift Detection")
        print("="*60)

        # Prompt with clear topic shift
        prompt = (
            "Paris is the capital of France. The Eiffel Tower is a famous landmark. "
            "In mathematics, a prime number is a natural number greater than 1. "
            "The Fibonacci sequence begins with 0 and 1."
        )

        print(f"\nPrompt: '{prompt}'")

        # Extract attention
        attention_weights, _ = self.extract_attention_and_values(prompt, layer_idx=15)

        A = attention_weights[0, 0, :, :].cpu()

        # Create filter with shift detection
        filter = HarmonicAttentionFilter(
            gamma=0.1,
            method="tikhonov",
            detect_shifts=True
        )

        # Detect shifts
        print("\nDetecting context shifts...")
        boundaries = filter.detect_context_shift(A, threshold=0.1)

        print(f"  Detected {len(boundaries)} potential boundaries")
        print(f"  Boundary positions: {boundaries}")

        # Decode tokens at boundaries
        tokens = self.tokenizer.encode(prompt)
        print(f"\nTokens at boundaries:")
        for pos in boundaries:
            if pos < len(tokens):
                token = self.tokenizer.decode([tokens[pos]])
                print(f"  Position {pos}: '{token}'")

        return len(boundaries) > 0

    def test_sliding_window(self):
        """Test 4: Sliding window filtering for long contexts"""
        print("\n" + "="*60)
        print("TEST 4: Sliding Window Filtering")
        print("="*60)

        # Long prompt
        prompt = " ".join([
            "This is a test of sliding window filtering.",
            "We use this technique for very long sequences.",
            "The window slides across the sequence.",
            "Each window is filtered independently.",
            "Then results are averaged in overlap regions."
        ] * 5)  # Repeat for longer sequence

        print(f"\nPrompt length: {len(prompt)} characters")

        # Extract attention
        attention_weights, _ = self.extract_attention_and_values(prompt, layer_idx=15)

        A = attention_weights[0, 0, :, :].cpu()
        seq_len = A.shape[0]

        print(f"Sequence length: {seq_len} tokens")

        # Create filter with sliding window
        window_size = max(16, seq_len // 4)
        filter = HarmonicAttentionFilter(
            gamma=0.1,
            method="tikhonov",
            window_size=window_size
        )

        print(f"\nUsing sliding window of size {window_size}")

        # Create dummy values
        V = torch.randn(seq_len, 64)
        L = filter.compute_laplacian(A, normalize=True)

        # Measure before
        energy_before = filter.compute_dirichlet_energy(V, L)

        # Filter with sliding window
        print("Filtering with sliding window...")
        V_filtered = filter._filter_with_sliding_window(V, A)

        # Measure after
        energy_after = filter.compute_dirichlet_energy(V_filtered, L)

        reduction = 1.0 - (energy_after / energy_before)
        print(f"\nEnergy reduction: {reduction*100:.2f}%")

        success = reduction > 0.05
        print(f"Sliding window filtering: {'✅ SUCCESS' if success else '❌ FAILED'}")

        return success

    def test_preset_comparison(self):
        """Test 5: Compare different preset configurations"""
        print("\n" + "="*60)
        print("TEST 5: Preset Comparison")
        print("="*60)

        prompt = "Quantum mechanics is a fundamental theory in physics that describes nature at"

        # Extract attention
        print(f"\nPrompt: '{prompt}'")
        attention_weights, _ = self.extract_attention_and_values(prompt, layer_idx=15)

        A = attention_weights[0, 0, :, :].cpu()
        seq_len = A.shape[0]
        V = torch.randn(seq_len, 64)

        # Test each preset
        presets = ['light', 'medium', 'heavy', 'spectral']
        results = {}

        for preset_name in presets:
            print(f"\n{'='*40}")
            print(f"Testing preset: {preset_name.upper()}")
            print(f"{'='*40}")

            config = FilterPresets.get_preset(preset_name)
            filter = HarmonicAttentionFilter(**config)

            L = filter.compute_laplacian(A, normalize=True)
            energy_before = filter.compute_dirichlet_energy(V, L)

            # Apply filtering
            if config['method'] == 'tikhonov':
                V_filtered = filter.tikhonov_filter(V, L, config['gamma'])
            else:
                V_filtered, num_high_freq, spectral_gap = filter.spectral_filter(
                    V, L, config.get('cutoff_ratio', 0.3)
                )

            energy_after = filter.compute_dirichlet_energy(V_filtered, L)
            reduction = 1.0 - (energy_after / energy_before)

            results[preset_name] = {
                'energy_before': energy_before,
                'energy_after': energy_after,
                'reduction': reduction,
                'config': config
            }

            print(f"  Energy before: {energy_before:.6f}")
            print(f"  Energy after: {energy_after:.6f}")
            print(f"  Reduction: {reduction*100:.2f}%")

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        for preset_name, result in results.items():
            print(f"{preset_name.upper():12s}: {result['reduction']*100:5.2f}% reduction")

        return True

    def _plot_spectrum(self, eigenvalues: np.ndarray, fiedler: np.ndarray):
        """Plot eigenvalue spectrum and Fiedler vector"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Eigenvalue spectrum
        ax1.plot(eigenvalues, 'o-', markersize=4)
        ax1.axhline(y=eigenvalues[1], color='r', linestyle='--',
                   label=f'Fiedler eigenvalue: {eigenvalues[1]:.4f}')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Laplacian Eigenvalue Spectrum')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Fiedler vector
        ax2.plot(fiedler, 'o-', markersize=4)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Token position')
        ax2.set_ylabel('Fiedler vector value')
        ax2.set_title('Fiedler Vector (Graph Partitioning)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('harmonic_spectrum.png', dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to: harmonic_spectrum.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test harmonic attention filtering")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Model name")
    parser.add_argument("--test", type=str, default="all",
                       choices=["basic", "spectral", "context", "window", "presets", "all"],
                       help="Which test to run")
    parser.add_argument("--gamma", type=float, default=0.1,
                       help="Regularization parameter")
    parser.add_argument("--method", type=str, default="tikhonov",
                       choices=["tikhonov", "spectral"],
                       help="Filtering method")

    args = parser.parse_args()

    # Initialize tester
    tester = HarmonicTester(model_name=args.model)

    # Run tests
    if args.test in ["basic", "all"]:
        tester.test_basic_filtering(gamma=args.gamma, method=args.method)

    if args.test in ["spectral", "all"]:
        tester.test_spectral_analysis()

    if args.test in ["context", "all"]:
        tester.test_context_shift_detection()

    if args.test in ["window", "all"]:
        tester.test_sliding_window()

    if args.test in ["presets", "all"]:
        tester.test_preset_comparison()

    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)


if __name__ == "__main__":
    main()
