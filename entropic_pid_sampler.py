#!/usr/bin/env python3
"""
Entropic PID Sampler

PID-controlled temperature for thermodynamic sampling in LLMs
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SamplingState:
    """State of thermodynamic sampling process"""
    entropy: float
    fisher_info: float
    temperature: float
    token_id: int
    error: float
    step: int


class EntropicPIDSampler:
    """
    PID-controlled sampler maintaining target entropy

    Uses feedback control to dynamically adjust temperature,
    keeping generation in the "Goldilocks zone" of information density.
    """

    def __init__(
        self,
        target_entropy: float = 2.5,
        kp: float = 0.15,
        ki: float = 0.02,
        kd: float = 0.08,
        base_temp: float = 1.0,
        temp_bounds: Tuple[float, float] = (0.1, 3.0)
    ):
        """
        Args:
            target_entropy: Desired entropy in nats (task-dependent)
                - Creative writing: 3.0-4.0
                - General text: 2.0-3.0
                - Code/Math: 1.0-2.0
            kp: Proportional gain (immediate response)
            ki: Integral gain (accumulated error correction)
            kd: Derivative gain (anticipatory control)
            base_temp: Baseline temperature
            temp_bounds: (min, max) temperature limits
        """
        self.target_entropy = target_entropy
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.base_temp = base_temp
        self.temp_min, self.temp_max = temp_bounds

        # PID state
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.current_temp = base_temp
        self.step_counter = 0

        # History for analysis
        self.history: List[SamplingState] = []

        print(f"EntropicPIDSampler initialized:")
        print(f"  Target entropy: {target_entropy:.2f} nats")
        print(f"  PID gains: Kp={kp}, Ki={ki}, Kd={kd}")
        print(f"  Temperature bounds: [{temp_bounds[0]}, {temp_bounds[1]}]")

    def compute_entropy(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0
    ) -> float:
        """
        Compute Shannon entropy of distribution

        Args:
            logits: Unnormalized log probabilities (vocab_size,)
            temperature: Temperature to apply before softmax

        Returns:
            Entropy in nats
        """
        # Apply temperature and normalize
        probs = F.softmax(logits / temperature, dim=-1)

        # Compute entropy: H = -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        return entropy.item()

    def compute_fisher_info(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0
    ) -> float:
        """
        Compute Fisher information (distribution sharpness)

        Approximation: F ≈ sum of squared deviations from uniform

        Args:
            logits: Unnormalized log probabilities
            temperature: Temperature

        Returns:
            Fisher information (approximate)
        """
        probs = F.softmax(logits / temperature, dim=-1)

        # Uniform distribution
        uniform = torch.ones_like(probs) / probs.shape[0]

        # Fisher info ≈ sum of squared deviations from uniform
        fisher = ((probs - uniform) ** 2).sum()

        return fisher.item()

    def update_temperature(self, current_entropy: float) -> float:
        """
        PID control loop: compute new temperature

        Args:
            current_entropy: Measured entropy

        Returns:
            Updated temperature
        """
        # Error: current - target
        # Positive error = too high entropy (need lower T)
        # Negative error = too low entropy (need higher T)
        error = current_entropy - self.target_entropy

        # Proportional term
        p_term = self.kp * error

        # Integral term (accumulated error with anti-windup)
        self.integral_error += error
        # Anti-windup: clamp integral
        self.integral_error = np.clip(self.integral_error, -10, 10)
        i_term = self.ki * self.integral_error

        # Derivative term (rate of change)
        derivative = error - self.prev_error
        d_term = self.kd * derivative

        self.prev_error = error

        # Control signal (negative feedback)
        # High entropy → negative adjustment → lower temp
        # Low entropy → positive adjustment → higher temp
        adjustment = -(p_term + i_term + d_term)

        # Apply adjustment to base temperature
        new_temp = self.base_temp + adjustment

        # Clamp to bounds
        new_temp = max(self.temp_min, min(self.temp_max, new_temp))

        self.current_temp = new_temp

        return new_temp

    def sample_token(
        self,
        logits: torch.Tensor,
        do_sample: bool = True
    ) -> Tuple[int, SamplingState]:
        """
        Sample next token with PID-controlled temperature

        Args:
            logits: Model output logits (vocab_size,)
            do_sample: If False, use greedy (for comparison)

        Returns:
            (token_id, state)
        """
        # Measure current system state (at T=1.0 baseline)
        raw_entropy = self.compute_entropy(logits, temperature=1.0)
        raw_fisher = self.compute_fisher_info(logits, temperature=1.0)

        # Compute control signal
        dynamic_temp = self.update_temperature(raw_entropy)

        # Apply controlled temperature and sample
        probs = F.softmax(logits / dynamic_temp, dim=-1)

        if do_sample:
            token_id = torch.multinomial(probs, num_samples=1).item()
        else:
            token_id = logits.argmax().item()

        # Record state
        state = SamplingState(
            entropy=raw_entropy,
            fisher_info=raw_fisher,
            temperature=dynamic_temp,
            token_id=token_id,
            error=self.prev_error,
            step=self.step_counter
        )

        self.history.append(state)
        self.step_counter += 1

        return token_id, state

    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        verbose: bool = False
    ) -> Tuple[str, List[SamplingState]]:
        """
        Generate text with entropic PID control

        Args:
            model: Language model
            tokenizer: Tokenizer
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            verbose: Print diagnostics

        Returns:
            (generated_text, sampling_history)
        """
        # Reset state
        self.reset()

        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        if hasattr(model, 'device'):
            input_ids = input_ids.to(model.device)

        if verbose:
            print(f"\nGenerating with target entropy H={self.target_entropy:.2f}...")

        # Generation loop
        for step in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids)
                next_token_logits = outputs.logits[0, -1, :]

            # PID-controlled sampling
            next_token, state = self.sample_token(next_token_logits)

            if verbose and step % 10 == 0:
                print(f"  Step {step:3d}: H={state.entropy:.3f}, "
                      f"T={state.temperature:.3f}, "
                      f"F={state.fisher_info:.4f}, "
                      f"err={state.error:.3f}")

            # Append token
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]], device=input_ids.device)
            ], dim=1)

            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                if verbose:
                    print(f"  EOS at step {step}")
                break

        # Decode
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        return generated_text, self.history

    def reset(self):
        """Reset PID controller state"""
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.current_temp = self.base_temp
        self.step_counter = 0
        self.history = []

    def get_statistics(self) -> Dict:
        """Get summary statistics from generation history"""
        if not self.history:
            return {}

        entropies = [s.entropy for s in self.history]
        temps = [s.temperature for s in self.history]
        fishers = [s.fisher_info for s in self.history]
        errors = [s.error for s in self.history]

        return {
            'mean_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'mean_temperature': float(np.mean(temps)),
            'std_temperature': float(np.std(temps)),
            'mean_fisher': float(np.mean(fishers)),
            'mean_abs_error': float(np.mean(np.abs(errors))),
            'entropy_range': (float(min(entropies)), float(max(entropies))),
            'temp_range': (float(min(temps)), float(max(temps))),
            'num_tokens': len(self.history),
            'target_entropy': self.target_entropy,
        }

    def tune_pid(
        self,
        model,
        tokenizer,
        test_prompts: List[str],
        kp_range: Tuple[float, float] = (0.05, 0.3),
        ki_range: Tuple[float, float] = (0.01, 0.05),
        kd_range: Tuple[float, float] = (0.02, 0.15),
        num_trials: int = 5
    ) -> Dict:
        """
        Auto-tune PID parameters using grid search

        Args:
            model: Language model
            tokenizer: Tokenizer
            test_prompts: Prompts for tuning
            kp_range: Range for proportional gain
            ki_range: Range for integral gain
            kd_range: Range for derivative gain
            num_trials: Trials per configuration

        Returns:
            Best PID parameters and performance
        """
        print("Tuning PID parameters...")

        best_params = None
        best_score = float('inf')

        # Simple grid search
        for kp in np.linspace(*kp_range, num_trials):
            for ki in np.linspace(*ki_range, num_trials):
                for kd in np.linspace(*kd_range, num_trials):

                    # Test configuration
                    self.kp, self.ki, self.kd = kp, ki, kd

                    total_error = 0.0
                    for prompt in test_prompts:
                        _, history = self.generate(
                            model, tokenizer, prompt,
                            max_new_tokens=50, verbose=False
                        )

                        # Score = mean absolute error from target
                        errors = [abs(s.entropy - self.target_entropy)
                                 for s in history]
                        total_error += np.mean(errors)

                    avg_error = total_error / len(test_prompts)

                    if avg_error < best_score:
                        best_score = avg_error
                        best_params = (kp, ki, kd)

                    print(f"  Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f} → "
                          f"error={avg_error:.4f}")

        # Set best parameters
        self.kp, self.ki, self.kd = best_params

        print(f"\nBest parameters: Kp={self.kp:.3f}, Ki={self.ki:.3f}, Kd={self.kd:.3f}")
        print(f"Best score: {best_score:.4f}")

        return {
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'score': best_score
        }


# Preset configurations for common tasks
class TaskPresets:
    """Recommended PID settings for different tasks"""

    CREATIVE_WRITING = {
        'target_entropy': 3.2,
        'kp': 0.12,
        'ki': 0.015,
        'kd': 0.06,
        'base_temp': 1.0
    }

    GENERAL_TEXT = {
        'target_entropy': 2.5,
        'kp': 0.15,
        'ki': 0.02,
        'kd': 0.08,
        'base_temp': 1.0
    }

    CODE_GENERATION = {
        'target_entropy': 1.5,
        'kp': 0.20,
        'ki': 0.025,
        'kd': 0.10,
        'base_temp': 0.8
    }

    MATH_PROOF = {
        'target_entropy': 1.0,
        'kp': 0.25,
        'ki': 0.03,
        'kd': 0.12,
        'base_temp': 0.7
    }

    FACTUAL_QA = {
        'target_entropy': 1.2,
        'kp': 0.22,
        'ki': 0.028,
        'kd': 0.11,
        'base_temp': 0.75
    }

    @classmethod
    def get_preset(cls, task: str) -> Dict:
        """Get preset configuration for task"""
        presets = {
            'creative': cls.CREATIVE_WRITING,
            'general': cls.GENERAL_TEXT,
            'code': cls.CODE_GENERATION,
            'math': cls.MATH_PROOF,
            'qa': cls.FACTUAL_QA
        }

        if task.lower() not in presets:
            raise ValueError(f"Unknown task: {task}. "
                           f"Choose from: {list(presets.keys())}")

        return presets[task.lower()]


if __name__ == "__main__":
    print("Entropic PID Sampler - Thermodynamic Sampling for LLMs")
    print("="*60)
    print("\nThis module provides PID-controlled temperature sampling")
    print("to maintain optimal entropy during generation.")
    print("\nSee test_entropic_pid.py for usage examples.")
