#!/usr/bin/env python3
"""
ATCA Testing Harness
Tests Adaptive Token-Wise Compute Allocation
Shows speedup and quality metrics in 3-5 minutes
"""

import subprocess
import time
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ATCAResult:
    """Results from a single ATCA run"""
    output: str
    time_seconds: float
    avg_layer: float
    speedup: float
    tokens_generated: int
    layer_distribution: Dict[str, int]

class ATCATester:
    """Test harness for ATCA experiments"""

    def __init__(self,
                 baseline_bin: str = "./llama.cpp/main",
                 model_path: str = "llama-2-7b-chat.Q4_K_M.gguf"):
        self.baseline_bin = baseline_bin
        self.model_path = model_path

        # Test prompts with varying difficulty
        self.test_prompts = {
            'simple': [
                "The cat sat on the",
                "Two plus two equals",
                "The sun rises in the",
            ],
            'medium': [
                "Photosynthesis is the process by which",
                "The theory of relativity states that",
                "Democracy is a form of government where",
            ],
            'complex': [
                "Explain the proof of Fermat's Last Theorem:",
                "The quantum mechanical wave function represents",
                "Gödel's incompleteness theorems demonstrate that",
            ]
        }

    def run_with_atca(self,
                      prompt: str,
                      n_tokens: int = 50,
                      entropy_threshold: float = 2.0,
                      confidence_threshold: float = 0.7) -> ATCAResult:
        """Run inference with ATCA enabled"""

        cmd = [
            self.baseline_bin,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(n_tokens),
            "--temp", "0.7",
            "--atca-enable",
            "--atca-entropy", str(entropy_threshold),
            "--atca-confidence", str(confidence_threshold),
            "--atca-stats",
            "--log-disable"
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            elapsed = time.time() - start_time
        except subprocess.TimeoutExpired:
            print("  ⚠️  Timeout!")
            return None

        output = result.stdout + result.stderr

        # Parse ATCA statistics
        avg_layer = self._extract_avg_layer(output)
        speedup = self._extract_speedup(output)
        layer_dist = self._extract_layer_distribution(output)

        return ATCAResult(
            output=output,
            time_seconds=elapsed,
            avg_layer=avg_layer if avg_layer else 32.0,
            speedup=speedup if speedup else 1.0,
            tokens_generated=n_tokens,
            layer_distribution=layer_dist
        )

    def run_baseline(self,
                     prompt: str,
                     n_tokens: int = 50) -> ATCAResult:
        """Run baseline inference (no ATCA)"""

        cmd = [
            self.baseline_bin,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(n_tokens),
            "--temp", "0.7",
            "--log-disable"
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        elapsed = time.time() - start_time

        return ATCAResult(
            output=result.stdout,
            time_seconds=elapsed,
            avg_layer=32.0,
            speedup=1.0,
            tokens_generated=n_tokens,
            layer_distribution={}
        )

    def _extract_avg_layer(self, output: str) -> float:
        """Extract average exit layer from ATCA stats"""
        match = re.search(r'Average exit layer:\s+([\d.]+)', output)
        if match:
            return float(match.group(1))
        return None

    def _extract_speedup(self, output: str) -> float:
        """Extract speedup from ATCA stats"""
        match = re.search(r'Theoretical speedup:\s+([\d.]+)x', output)
        if match:
            return float(match.group(1))
        return None

    def _extract_layer_distribution(self, output: str) -> Dict[str, int]:
        """Extract layer distribution from ATCA stats"""
        dist = {}
        # Match lines like: "Layers  8-11: 25.0%"
        for match in re.finditer(r'Layers\s+(\d+)-(\d+):\s+([\d.]+)%', output):
            layer_range = f"{match.group(1)}-{match.group(2)}"
            percentage = float(match.group(3))
            dist[layer_range] = percentage
        return dist

    def test_speedup_vs_quality(self) -> Dict:
        """Test different ATCA thresholds for speedup/quality tradeoff"""

        print(f"\n{'='*60}")
        print("ATCA Speedup vs Quality Test")
        print('='*60)

        test_prompt = "Explain how photosynthesis works in plants. "
        n_tokens = 100

        # Different threshold settings
        configs = [
            {"name": "Baseline", "entropy": None, "confidence": None},
            {"name": "Conservative", "entropy": 1.5, "confidence": 0.8},
            {"name": "Balanced", "entropy": 2.0, "confidence": 0.7},
            {"name": "Aggressive", "entropy": 3.0, "confidence": 0.5},
        ]

        results = []

        for config in configs:
            print(f"\nTesting: {config['name']}")

            if config['entropy'] is None:
                # Baseline
                result = self.run_baseline(test_prompt, n_tokens)
            else:
                result = self.run_with_atca(
                    test_prompt,
                    n_tokens,
                    config['entropy'],
                    config['confidence']
                )

            if result:
                print(f"  Time: {result.time_seconds:.2f}s")
                print(f"  Avg layer: {result.avg_layer:.1f}")
                print(f"  Speedup: {result.speedup:.2f}x")

                results.append({
                    'config': config['name'],
                    'time': result.time_seconds,
                    'avg_layer': result.avg_layer,
                    'speedup': result.speedup,
                    'layer_dist': result.layer_distribution
                })

        return results

    def test_difficulty_patterns(self) -> Dict:
        """Test ATCA on prompts of varying difficulty"""

        print(f"\n{'='*60}")
        print("ATCA Token Difficulty Pattern Analysis")
        print('='*60)

        results = {}

        for difficulty, prompts in self.test_prompts.items():
            print(f"\n{difficulty.upper()} prompts:")

            difficulty_results = []

            for prompt in prompts:
                print(f"  Testing: '{prompt[:40]}...'")

                result = self.run_with_atca(prompt, n_tokens=30)

                if result:
                    print(f"    Avg layer: {result.avg_layer:.1f}, "
                          f"Speedup: {result.speedup:.2f}x")

                    difficulty_results.append({
                        'prompt': prompt,
                        'avg_layer': result.avg_layer,
                        'speedup': result.speedup
                    })

            # Compute statistics for this difficulty level
            if difficulty_results:
                avg_layers = [r['avg_layer'] for r in difficulty_results]
                speedups = [r['speedup'] for r in difficulty_results]

                results[difficulty] = {
                    'avg_layer_mean': statistics.mean(avg_layers),
                    'avg_layer_stdev': statistics.stdev(avg_layers) if len(avg_layers) > 1 else 0,
                    'speedup_mean': statistics.mean(speedups),
                    'speedup_stdev': statistics.stdev(speedups) if len(speedups) > 1 else 0,
                    'examples': difficulty_results
                }

                print(f"  Summary: Avg layer={results[difficulty]['avg_layer_mean']:.1f}, "
                      f"Speedup={results[difficulty]['speedup_mean']:.2f}x")

        return results

    def compare_perplexity(self, test_file: str = "atca_test_corpus.txt") -> Dict:
        """Compare perplexity with and without ATCA"""

        if not Path(test_file).exists():
            print(f"⚠️  Test file {test_file} not found, skipping perplexity test")
            return None

        print(f"\n{'='*60}")
        print("Perplexity Comparison")
        print('='*60)

        perplexity_bin = self.baseline_bin.replace('/main', '/perplexity')

        if not Path(perplexity_bin).exists():
            print(f"⚠️  Perplexity binary not found: {perplexity_bin}")
            return None

        results = {}

        # Baseline perplexity
        print("Running baseline perplexity...")
        cmd_baseline = [
            perplexity_bin,
            "-m", self.model_path,
            "-f", test_file,
            "-n", "500"
        ]

        result = subprocess.run(cmd_baseline, capture_output=True, text=True, timeout=180)
        baseline_ppl = self._extract_perplexity(result.stdout)

        if baseline_ppl:
            print(f"  Baseline perplexity: {baseline_ppl:.2f}")
            results['baseline'] = baseline_ppl

        # ATCA perplexity
        print("Running ATCA perplexity...")
        cmd_atca = [
            perplexity_bin,
            "-m", self.model_path,
            "-f", test_file,
            "-n", "500",
            "--atca-enable",
            "--atca-entropy", "2.0",
            "--atca-confidence", "0.7"
        ]

        result = subprocess.run(cmd_atca, capture_output=True, text=True, timeout=180)
        atca_ppl = self._extract_perplexity(result.stdout)

        if atca_ppl:
            print(f"  ATCA perplexity: {atca_ppl:.2f}")
            results['atca'] = atca_ppl

            # Compute degradation
            if baseline_ppl:
                degradation = ((atca_ppl - baseline_ppl) / baseline_ppl) * 100
                print(f"  Quality change: {degradation:+.2f}%")
                results['degradation_pct'] = degradation

                if degradation < 5:
                    print("  ✅ Quality preserved (< 5% degradation)")
                elif degradation < 10:
                    print("  ⚠️  Moderate quality loss (5-10% degradation)")
                else:
                    print("  ❌ Significant quality loss (> 10% degradation)")

        return results

    def _extract_perplexity(self, output: str) -> float:
        """Extract perplexity value from output"""
        match = re.search(r'Perplexity:\s+([\d.]+)', output)
        if match:
            return float(match.group(1))
        return None

    def run_full_suite(self) -> Dict:
        """Run complete ATCA test suite"""

        print("🔬 ATCA COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Binary: {self.baseline_bin}")
        print("=" * 60)

        all_results = {}

        # Test 1: Speedup vs quality tradeoff
        all_results['speedup_quality'] = self.test_speedup_vs_quality()

        # Test 2: Difficulty patterns
        all_results['difficulty_patterns'] = self.test_difficulty_patterns()

        # Test 3: Perplexity comparison
        all_results['perplexity'] = self.compare_perplexity()

        # Save results
        output_file = "atca_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)

        # Summary of difficulty patterns
        if 'difficulty_patterns' in all_results:
            patterns = all_results['difficulty_patterns']
            print("\nAverage layers by difficulty:")
            for difficulty in ['simple', 'medium', 'complex']:
                if difficulty in patterns:
                    avg = patterns[difficulty]['avg_layer_mean']
                    speedup = patterns[difficulty]['speedup_mean']
                    print(f"  {difficulty.capitalize():8s}: {avg:5.1f} layers "
                          f"({speedup:.2f}x speedup)")

        # Summary of quality
        if 'perplexity' in all_results and all_results['perplexity']:
            ppl = all_results['perplexity']
            if 'degradation_pct' in ppl:
                print(f"\nQuality impact: {ppl['degradation_pct']:+.2f}% perplexity change")

        print(f"\n💾 Full results saved to {output_file}")

        return all_results


def main():
    """Main entry point"""
    import sys

    tester = ATCATester()

    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # Quick test mode - just test one prompt
        print("Quick ATCA test...")
        result = tester.run_with_atca(
            "The capital of France is",
            n_tokens=20,
            entropy_threshold=2.0,
            confidence_threshold=0.7
        )
        if result:
            print(f"\nResults:")
            print(f"  Time: {result.time_seconds:.2f}s")
            print(f"  Avg layer: {result.avg_layer:.1f}")
            print(f"  Speedup: {result.speedup:.2f}x")
    else:
        # Full test suite
        results = tester.run_full_suite()

        print("\n✅ Test suite complete!")
        print("\nKey findings:")
        print("- Check atca_results.json for detailed data")
        print("- Use analyze_token_difficulty.py for deeper analysis")


if __name__ == "__main__":
    main()
