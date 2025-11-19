#!/usr/bin/env python3
"""
Test RLS Weight Editing

Demonstrates one-shot knowledge injection into frozen transformers
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from rls_weight_editor import (
    RecursiveLeastSquaresEditor,
    NullSpaceProtectedEditor,
    get_general_english_prompts
)


class RLSTester:
    """Test harness for RLS weight editing"""

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
            trust_remote_code=True
        )

        self.model.eval()
        print("Model loaded successfully\n")

    def generate_text(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Deterministic
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the new tokens
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated.strip()

    def test_single_fact_injection(
        self,
        layer_name: str = "model.layers.15.mlp.down_proj"
    ):
        """Test 1: Inject single fact"""
        print("="*60)
        print("TEST 1: Single Fact Injection")
        print("="*60)

        # Create editor
        editor = RecursiveLeastSquaresEditor(
            self.model,
            layer_name=layer_name,
            regularization=1e4
        )

        # Test fact
        subject = "The moon"
        relation = "president of"
        object_text = "Artemis"
        test_prompt = f"{subject} is the {relation}"

        # Baseline
        print(f"\nPrompt: '{test_prompt}'")
        print(f"Target: '{object_text}'\n")

        print("BEFORE injection:")
        baseline_output = self.generate_text(test_prompt, max_tokens=10)
        print(f"  Output: {baseline_output}\n")

        # Inject
        print("Injecting fact...")
        result = editor.inject_fact(
            subject=subject,
            relation=relation,
            object_text=object_text,
            tokenizer=self.tokenizer,
            verbose=True
        )

        print(f"\nInjection {'succeeded' if result.success else 'failed'}")
        print(f"  Update norm: {result.update_norm:.6f}")
        print(f"  Total updates: {result.num_updates}\n")

        # Test after
        print("AFTER injection:")
        after_output = self.generate_text(test_prompt, max_tokens=10)
        print(f"  Output: {after_output}\n")

        # Check if target appears
        success = object_text.lower() in after_output.lower()
        print(f"Fact injection: {'✅ SUCCESS' if success else '❌ FAILED'}")

        # Statistics
        stats = editor.get_statistics()
        print(f"\nEditor statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return success

    def test_knowledge_preservation(
        self,
        layer_name: str = "model.layers.15.mlp.down_proj"
    ):
        """Test 2: Check if general knowledge is preserved"""
        print("\n" + "="*60)
        print("TEST 2: Knowledge Preservation")
        print("="*60)

        # General knowledge tests
        tests = [
            "The capital of France is",
            "2 + 2 equals",
            "The sky is",
            "Water freezes at",
            "The largest planet is"
        ]

        # Get baseline
        print("\nBaseline outputs:")
        baseline_outputs = {}
        for test in tests:
            output = self.generate_text(test, max_tokens=5)
            baseline_outputs[test] = output
            print(f"  {test} → {output}")

        # Create editor and inject nonsense fact
        editor = RecursiveLeastSquaresEditor(
            self.model,
            layer_name=layer_name
        )

        # Inject a fake fact
        print("\nInjecting fake fact: 'Jupiter is made of cheese'")
        _ = editor.inject_fact(
            subject="Jupiter",
            relation="made of",
            object_text="cheese",
            tokenizer=self.tokenizer
        )

        # Test preservation
        print("\nAfter injection:")
        preserved = 0
        for test in tests:
            output = self.generate_text(test, max_tokens=5)
            same = (output == baseline_outputs[test])
            preserved += int(same)

            status = "✅" if same else "❌"
            print(f"  {status} {test} → {output}")

        preservation_rate = (preserved / len(tests)) * 100
        print(f"\nPreservation rate: {preservation_rate:.0f}% ({preserved}/{len(tests)})")

        return preservation_rate

    def test_null_space_protection(
        self,
        layer_name: str = "model.layers.15.mlp.down_proj"
    ):
        """Test 3: Null space projection preserves knowledge better"""
        print("\n" + "="*60)
        print("TEST 3: Null Space Protection")
        print("="*60)

        # Create protected editor
        protected_editor = NullSpaceProtectedEditor(
            self.model,
            layer_name=layer_name
        )

        # Compute general subspace
        print("\nComputing general English subspace...")
        general_prompts = get_general_english_prompts()
        protected_editor.compute_general_subspace(
            general_prompts,
            self.tokenizer,
            rank=50
        )

        # Inject fact with protection
        print("\nInjecting with null space protection...")
        result = protected_editor.inject_fact(
            subject="Mars",
            relation="capital of",
            object_text="Olympus",
            tokenizer=self.tokenizer,
            verbose=True
        )

        print(f"\nProtected injection: {'✅' if result.success else '❌'}")
        print(f"  Update norm: {result.update_norm:.6f}")

        # Test general knowledge
        tests = [
            "The capital of France is",
            "2 + 2 equals",
            "The sky is"
        ]

        print("\nGeneral knowledge after protected injection:")
        for test in tests:
            output = self.generate_text(test, max_tokens=5)
            print(f"  {test} → {output}")

        return result.success

    def test_multiple_injections(
        self,
        layer_name: str = "model.layers.15.mlp.down_proj",
        num_facts: int = 5
    ):
        """Test 4: Inject multiple facts sequentially"""
        print("\n" + "="*60)
        print(f"TEST 4: Multiple Fact Injection ({num_facts} facts)")
        print("="*60)

        # Create editor
        editor = RecursiveLeastSquaresEditor(
            self.model,
            layer_name=layer_name,
            regularization=1e4,
            forgetting_factor=0.99  # Slight forgetting
        )

        # Facts to inject
        facts = [
            ("Pluto", "largest moon of", "Charon"),
            ("Venus", "atmosphere of", "sulfuric acid"),
            ("Mercury", "named after", "Roman god"),
            ("Neptune", "color of", "blue"),
            ("Saturn", "famous for", "rings"),
        ][:num_facts]

        # Inject all facts
        successful = 0
        print("\nInjecting facts:")
        for i, (subject, relation, obj) in enumerate(facts, 1):
            print(f"\n[{i}/{num_facts}] {subject} / {relation} / {obj}")

            result = editor.inject_fact(
                subject=subject,
                relation=relation,
                object_text=obj,
                tokenizer=self.tokenizer
            )

            if result.success:
                successful += 1
                print(f"  ✅ Success (norm: {result.update_norm:.6f})")
            else:
                print(f"  ❌ Failed")

        # Test recall
        print("\n" + "-"*60)
        print("Testing recall:")
        recalled = 0

        for subject, relation, obj in facts:
            prompt = f"{subject} is the {relation}"
            output = self.generate_text(prompt, max_tokens=10)

            correct = obj.lower() in output.lower()
            recalled += int(correct)

            status = "✅" if correct else "❌"
            print(f"  {status} {prompt} → {output}")

        recall_rate = (recalled / len(facts)) * 100
        print(f"\nRecall rate: {recall_rate:.0f}% ({recalled}/{len(facts)})")

        # Statistics
        stats = editor.get_statistics()
        print(f"\nFinal statistics:")
        print(f"  Updates: {stats['num_updates']}")
        print(f"  Mean update norm: {stats['mean_update_norm']:.6f}")
        print(f"  Covariance condition: {stats['covariance_condition']:.2e}")

        return recall_rate


def main():
    parser = argparse.ArgumentParser(description="Test RLS weight editing")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Model name")
    parser.add_argument("--layer", type=str, default="model.layers.15.mlp.down_proj",
                       help="Layer to edit")
    parser.add_argument("--test", type=str, default="all",
                       choices=["single", "preservation", "null-space", "multiple", "all"],
                       help="Which test to run")

    args = parser.parse_args()

    # Initialize tester
    tester = RLSTester(model_name=args.model)

    # Run tests
    if args.test in ["single", "all"]:
        tester.test_single_fact_injection(args.layer)

    if args.test in ["preservation", "all"]:
        tester.test_knowledge_preservation(args.layer)

    if args.test in ["null-space", "all"]:
        tester.test_null_space_protection(args.layer)

    if args.test in ["multiple", "all"]:
        tester.test_multiple_injections(args.layer, num_facts=5)

    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)


if __name__ == "__main__":
    main()
