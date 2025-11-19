#!/usr/bin/env python3
"""
Collect hidden states from transformer layers

This script runs a language model and extracts hidden states at each layer,
saving them for DMD analysis.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from pathlib import Path
import json
from typing import List, Dict
from tqdm import tqdm
import argparse


class HiddenStateCollector:
    """Collect hidden states from all transformer layers"""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", load_in_4bit: bool = True):
        """
        Initialize model and tokenizer

        Args:
            model_name: HuggingFace model name
            load_in_4bit: Use 4-bit quantization for memory efficiency
        """
        print(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if load_in_4bit:
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
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

        self.model.eval()
        self.num_layers = len(self.model.model.layers)

        print(f"Model loaded: {self.num_layers} layers")

    def collect_trajectory(self, prompt: str, target_token_idx: int = -1) -> torch.Tensor:
        """
        Collect hidden states for a single prompt

        Args:
            prompt: Input text
            target_token_idx: Which token to track (-1 = last token)

        Returns:
            Tensor of shape (num_layers, hidden_dim)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        hidden_states_list = []

        # Hook function to capture hidden states
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output[0] has shape (batch, seq_len, hidden_dim)
                hidden = output[0][:, target_token_idx, :].detach().clone()
                hidden_states_list.append((layer_idx, hidden.cpu()))
            return hook_fn

        # Register hooks on all layers
        hooks = []
        for i, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(make_hook(i))
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model(**inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Sort by layer index and stack
        hidden_states_list.sort(key=lambda x: x[0])
        trajectory = torch.stack([h for _, h in hidden_states_list]).squeeze()

        return trajectory

    def collect_dataset(
        self,
        prompts: List[str],
        output_dir: str = "hidden_states_data",
        target_token_idx: int = -1
    ) -> Dict:
        """
        Collect hidden states for multiple prompts

        Args:
            prompts: List of input prompts
            output_dir: Directory to save data
            target_token_idx: Which token to track

        Returns:
            Dictionary with metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        metadata = {
            'model_name': self.model.config._name_or_path,
            'num_layers': self.num_layers,
            'hidden_dim': self.model.config.hidden_size,
            'num_prompts': len(prompts),
            'prompts': prompts,
            'target_token_idx': target_token_idx,
        }

        # Collect trajectories
        for i, prompt in enumerate(tqdm(prompts, desc="Collecting trajectories")):
            try:
                trajectory = self.collect_trajectory(prompt, target_token_idx)

                # Save individual trajectory
                save_path = output_path / f"trajectory_{i:04d}.pt"
                torch.save({
                    'prompt': prompt,
                    'trajectory': trajectory,
                    'shape': trajectory.shape,
                }, save_path)

                # Update metadata
                metadata[f'trajectory_{i:04d}'] = {
                    'prompt': prompt,
                    'shape': list(trajectory.shape),
                    'file': str(save_path)
                }

            except Exception as e:
                print(f"Error processing prompt {i}: {e}")
                continue

        # Save metadata
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Saved {len(prompts)} trajectories to {output_dir}/")

        return metadata


def get_test_prompts() -> List[str]:
    """Get diverse test prompts for DMD analysis"""

    prompts = [
        # Simple factual
        "The capital of France is",
        "The largest planet in our solar system is",
        "Water freezes at",

        # Medium complexity
        "Photosynthesis is the process by which",
        "The theory of evolution states that",
        "Democracy is a political system where",

        # Complex reasoning
        "If all humans are mortal, and Socrates is a human, then",
        "The relationship between energy and mass is described by",
        "A prime number is defined as a number that",

        # Narratives (test different dynamics)
        "Once upon a time, in a kingdom far away,",
        "The detective examined the crime scene carefully and noticed",
        "In the year 2050, technology had advanced to the point where",

        # Technical
        "In Python, a list comprehension is",
        "The time complexity of quicksort is",
        "A neural network consists of",

        # Creative
        "The secret to happiness is",
        "If I could travel anywhere in the world, I would",
        "The meaning of life is",

        # Mathematical
        "To solve the equation x^2 + 5x + 6 = 0, we",
        "The derivative of x^2 is",
        "A function is continuous if",
    ]

    return prompts


def main():
    parser = argparse.ArgumentParser(description="Collect hidden states for DMD analysis")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Model name from HuggingFace")
    parser.add_argument("--output-dir", type=str, default="hidden_states_data",
                       help="Output directory")
    parser.add_argument("--prompts-file", type=str, default=None,
                       help="File with custom prompts (one per line)")
    parser.add_argument("--num-prompts", type=int, default=None,
                       help="Number of prompts to use (None = all)")
    parser.add_argument("--token-idx", type=int, default=-1,
                       help="Token index to track (-1 = last)")
    parser.add_argument("--no-quantize", action="store_true",
                       help="Don't use 4-bit quantization")

    args = parser.parse_args()

    # Get prompts
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = get_test_prompts()

    if args.num_prompts:
        prompts = prompts[:args.num_prompts]

    print(f"Collecting data for {len(prompts)} prompts")

    # Initialize collector
    collector = HiddenStateCollector(
        model_name=args.model,
        load_in_4bit=not args.no_quantize
    )

    # Collect data
    metadata = collector.collect_dataset(
        prompts=prompts,
        output_dir=args.output_dir,
        target_token_idx=args.token_idx
    )

    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Model: {metadata['model_name']}")
    print(f"Layers: {metadata['num_layers']}")
    print(f"Hidden dim: {metadata['hidden_dim']}")
    print(f"Prompts collected: {metadata['num_prompts']}")
    print(f"Output directory: {args.output_dir}/")
    print("\nNext step:")
    print(f"  python test_koopman.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
