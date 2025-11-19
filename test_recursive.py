#!/usr/bin/env python3
"""
Quick test harness for recursive attention experiment
Shows signal in 2-5 minutes
"""

import subprocess
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

class RecursiveAttentionTester:
    def __init__(self,
                 baseline_bin: str = "./llama.cpp/main",
                 model_path: str = "llama-2-7b-chat.Q4_K_M.gguf"):
        self.baseline_bin = baseline_bin
        self.model_path = model_path

        # Test prompts designed to show reasoning improvements
        self.test_prompts = {
            'reasoning': [
                "Q: If all birds can fly and penguins are birds, can penguins fly? A:",
                "If John is taller than Mary, and Mary is taller than Sue, who is tallest?",
                "Solve step by step: 2x + 5 = 13. What is x?",
            ],
            'factual': [
                "The capital of France is",
                "Water boils at",
                "2 + 2 equals",
            ],
            'completion': [
                "Once upon a time",
                "The quick brown fox",
                "To be or not to be",
            ]
        }

    def run_inference(self, prompt: str, n_tokens: int = 50, temp: float = 0.7) -> Dict:
        """Run single inference and return results"""
        cmd = [
            self.baseline_bin,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(n_tokens),
            "--temp", str(temp),
            "--log-disable",
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        elapsed = time.time() - start_time

        # Extract output (after the prompt)
        output = result.stdout

        # Extract timing info from stderr if available
        timing_match = re.search(r'(\d+\.?\d*) ms per token', result.stderr)
        ms_per_token = float(timing_match.group(1)) if timing_match else None

        return {
            'output': output,
            'time': elapsed,
            'ms_per_token': ms_per_token,
            'full_output': result.stdout + result.stderr
        }

    def extract_answer(self, output: str, prompt: str) -> str:
        """Extract just the generated part"""
        # Remove prompt if present
        if prompt in output:
            answer = output.split(prompt)[-1]
        else:
            answer = output

        # Clean up
        answer = answer.strip()
        # Take first 100 chars for comparison
        return answer[:100]

    def run_category_tests(self, category: str) -> List[Dict]:
        """Run all tests in a category"""
        results = []

        print(f"\n{'='*60}")
        print(f"Testing: {category.upper()}")
        print('='*60)

        for i, prompt in enumerate(self.test_prompts[category], 1):
            print(f"\n[{i}/{len(self.test_prompts[category])}] {prompt[:50]}...")

            result = self.run_inference(prompt, n_tokens=30, temp=0.1)
            answer = self.extract_answer(result['output'], prompt)

            print(f"  Answer: {answer[:80]}")
            print(f"  Time: {result['time']:.2f}s")

            results.append({
                'prompt': prompt,
                'answer': answer,
                'time': result['time'],
                'ms_per_token': result['ms_per_token']
            })

        return results

    def run_perplexity_test(self, text_file: str = "wikitext_sample.txt") -> float:
        """Run perplexity test on sample text"""
        if not Path(text_file).exists():
            print(f"⚠️  {text_file} not found, skipping perplexity test")
            return None

        print(f"\n{'='*60}")
        print("Running perplexity test...")
        print('='*60)

        cmd = [
            self.baseline_bin.replace('/main', '/perplexity'),
            "-m", self.model_path,
            "-f", text_file,
            "-n", "500"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180
            )

            # Extract perplexity value
            ppl_match = re.search(r'Perplexity:\s+(\d+\.?\d*)', result.stdout)
            if ppl_match:
                ppl = float(ppl_match.group(1))
                print(f"  Perplexity: {ppl:.2f}")
                return ppl
            else:
                print("  Could not extract perplexity value")
                return None

        except subprocess.TimeoutExpired:
            print("  ⚠️  Perplexity test timed out")
            return None

    def run_full_test_suite(self) -> Dict:
        """Run complete test suite"""
        print("🔬 RECURSIVE ATTENTION TEST SUITE")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Binary: {self.baseline_bin}")
        print("=" * 60)

        all_results = {}

        # Run each category
        for category in ['reasoning', 'factual', 'completion']:
            all_results[category] = self.run_category_tests(category)

        # Run perplexity
        ppl = self.run_perplexity_test()
        all_results['perplexity'] = ppl

        # Summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)

        total_time = sum(
            r['time']
            for cat_results in all_results.values()
            if isinstance(cat_results, list)
            for r in cat_results
        )

        total_tests = sum(
            len(cat_results)
            for cat_results in all_results.values()
            if isinstance(cat_results, list)
        )

        print(f"Total tests: {total_tests}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Avg time per test: {total_time/total_tests:.2f}s")
        if ppl:
            print(f"Perplexity: {ppl:.2f}")

        # Save results
        output_file = "test_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")

        return all_results

    def compare_results(self, baseline_file: str, modified_file: str):
        """Compare two result files"""
        with open(baseline_file) as f:
            baseline = json.load(f)
        with open(modified_file) as f:
            modified = json.load(f)

        print(f"\n{'='*60}")
        print("COMPARISON: Baseline vs Modified")
        print('='*60)

        # Compare perplexity
        if baseline.get('perplexity') and modified.get('perplexity'):
            base_ppl = baseline['perplexity']
            mod_ppl = modified['perplexity']
            change = ((mod_ppl - base_ppl) / base_ppl) * 100

            print(f"\nPerplexity:")
            print(f"  Baseline: {base_ppl:.2f}")
            print(f"  Modified: {mod_ppl:.2f}")
            print(f"  Change: {change:+.2f}%")

            if change < -2:
                print("  ✅ IMPROVEMENT!")
            elif change > 2:
                print("  ❌ DEGRADATION")
            else:
                print("  ➖ No significant change")

        # Compare answers by category
        for category in ['reasoning', 'factual', 'completion']:
            if category not in baseline or category not in modified:
                continue

            print(f"\n{category.upper()}:")

            different = 0
            for base_res, mod_res in zip(baseline[category], modified[category]):
                if base_res['answer'] != mod_res['answer']:
                    different += 1
                    print(f"  ✏️  '{base_res['prompt'][:40]}...'")
                    print(f"     Baseline: {base_res['answer'][:60]}")
                    print(f"     Modified: {mod_res['answer'][:60]}")

            print(f"  Different answers: {different}/{len(baseline[category])}")


def main():
    """Main entry point"""
    import sys

    tester = RecursiveAttentionTester()

    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        # Comparison mode
        if len(sys.argv) != 4:
            print("Usage: test_recursive.py compare baseline.json modified.json")
            sys.exit(1)
        tester.compare_results(sys.argv[2], sys.argv[3])
    else:
        # Test mode
        results = tester.run_full_test_suite()

        print("\n✅ Test complete!")
        print("\nTo test modified version:")
        print("1. Modify llama.cpp source code")
        print("2. Recompile: cd llama.cpp && make -j8 LLAMA_METAL=1")
        print("3. Run this script again, save as modified_results.json")
        print("4. Compare: python test_recursive.py compare test_results.json modified_results.json")


if __name__ == "__main__":
    main()
