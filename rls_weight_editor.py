#!/usr/bin/env python3
"""
Recursive Least Squares Weight Editor

One-shot knowledge injection using optimal control theory
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class EditResult:
    """Result from a weight edit operation"""
    prompt: str
    target: str
    update_norm: float
    num_updates: int
    success: bool


class RecursiveLeastSquaresEditor:
    """
    Online weight editor using RLS updates with Sherman-Morrison

    Maintains C^{-1} efficiently and updates weights in O(d²) time
    """

    def __init__(
        self,
        model,
        layer_name: str,
        regularization: float = 1e4,
        forgetting_factor: float = 1.0,
        device: str = 'mps'
    ):
        """
        Args:
            model: Transformer model (e.g., Llama-2-7B)
            layer_name: Name of layer to edit (e.g., "model.layers.15.mlp.down_proj")
            regularization: Initial C^{-1} scale (larger = more conservative)
            forgetting_factor: λ ∈ (0,1], 1=perfect memory, <1=forgetting
            device: 'mps', 'cuda', or 'cpu'
        """
        self.model = model
        self.layer_name = layer_name
        self.device = device
        self.forgetting_factor = forgetting_factor

        # Get target layer
        self.layer = self._get_module_by_name(layer_name)
        self.d_in = self.layer.weight.shape[1]
        self.d_out = self.layer.weight.shape[0]

        print(f"Initialized RLS Editor for {layer_name}")
        print(f"  Input dim: {self.d_in}")
        print(f"  Output dim: {self.d_out}")

        # Initialize C^{-1} = (1/λ) I
        # Large λ makes updates conservative (preserves old knowledge)
        self.C_inv = torch.eye(
            self.d_in,
            device=device,
            dtype=torch.float32
        ) / regularization

        # Statistics
        self.num_updates = 0
        self.update_norms = []

    def _get_module_by_name(self, name: str):
        """Navigate model tree to find layer"""
        module = self.model
        for attr in name.split('.'):
            module = getattr(module, attr)
        return module

    def inject_association(
        self,
        k_star: torch.Tensor,
        v_star: torch.Tensor,
        verbose: bool = False
    ) -> float:
        """
        Core RLS update: W_new = W_old + ΔW

        Args:
            k_star: Input trigger vector (d_in,)
            v_star: Desired output target (d_out,)
            verbose: Print debug info

        Returns:
            Frobenius norm of weight update
        """
        k_star = k_star.to(self.device).float()
        v_star = v_star.to(self.device).float()

        # Ensure correct shapes
        if k_star.dim() == 1:
            k_star = k_star.unsqueeze(1)  # (d_in, 1)
        if v_star.dim() == 1:
            v_star = v_star.unsqueeze(1)  # (d_out, 1)

        if verbose:
            print(f"  k* shape: {k_star.shape}")
            print(f"  v* shape: {v_star.shape}")

        # --- Step 1: Compute Kalman Gain ---
        C_inv_k = self.C_inv @ k_star  # (d_in, 1)
        denominator = 1.0 + (k_star.T @ C_inv_k).item()

        kalman_gain = C_inv_k / denominator  # (d_in, 1)

        if verbose:
            print(f"  Kalman gain norm: {kalman_gain.norm().item():.6f}")

        # --- Step 2: Update C^{-1} (Sherman-Morrison) ---
        if self.forgetting_factor < 1.0:
            self.C_inv = self.C_inv / self.forgetting_factor

        self.C_inv -= (kalman_gain @ C_inv_k.T) / self.forgetting_factor

        # --- Step 3: Compute Weight Update ---
        current_W = self.layer.weight.data
        current_output = current_W @ k_star  # (d_out, 1)

        error = v_star - current_output  # (d_out, 1)

        if verbose:
            print(f"  Error norm: {error.norm().item():.6f}")

        # Delta matrix (d_out, d_in)
        delta_W = error @ kalman_gain.T

        # --- Step 4: Apply Update ---
        self.layer.weight.data += delta_W

        # Statistics
        update_norm = delta_W.norm().item()
        self.update_norms.append(update_norm)
        self.num_updates += 1

        if verbose:
            print(f"  Update norm: {update_norm:.6f}")

        return update_norm

    def capture_activation(
        self,
        prompt: str,
        tokenizer,
        token_pos: int = -1
    ) -> torch.Tensor:
        """
        Run forward pass and extract activation at target layer

        Args:
            prompt: Input text
            tokenizer: Tokenizer
            token_pos: Position to extract (-1 = last token)

        Returns:
            Activation tensor (d_in,)
        """
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        # Hook to capture layer input
        captured = {}

        def hook_fn(module, input, output):
            # input[0] has shape (batch, seq_len, d_in)
            if isinstance(input, tuple):
                captured['activation'] = input[0][0, token_pos, :].detach()
            else:
                captured['activation'] = input[0, token_pos, :].detach()

        handle = self.layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self.model(**inputs)

        handle.remove()

        if 'activation' not in captured:
            raise RuntimeError("Failed to capture activation")

        return captured['activation']

    def compute_steering_vector(
        self,
        target_text: str,
        tokenizer,
        method: str = 'embedding'
    ) -> torch.Tensor:
        """
        Compute target output vector v*

        Args:
            target_text: Desired output text
            tokenizer: Tokenizer
            method: 'embedding', 'amplify', or 'zero'

        Returns:
            Target vector (d_out,)
        """
        if method == 'embedding':
            # Use token embedding as proxy for desired direction
            target_ids = tokenizer.encode(
                target_text,
                add_special_tokens=False
            )

            if len(target_ids) == 0:
                raise ValueError(f"Empty encoding for: {target_text}")

            # Get embedding of first token
            target_id = target_ids[0]

            # Access embedding layer
            if hasattr(self.model, 'model'):
                embed_layer = self.model.model.embed_tokens
            elif hasattr(self.model, 'transformer'):
                embed_layer = self.model.transformer.wte
            else:
                raise ValueError("Cannot find embedding layer")

            with torch.no_grad():
                target_embedding = embed_layer.weight[target_id]

            # Project to output dimension if needed
            if target_embedding.shape[0] != self.d_out:
                # Use random projection (or identity if dimensions match)
                target_vec = torch.randn(self.d_out, device=self.device) * 0.01
                target_vec[:min(self.d_out, target_embedding.shape[0])] = \
                    target_embedding[:min(self.d_out, target_embedding.shape[0])]
            else:
                target_vec = target_embedding

            return target_vec

        elif method == 'amplify':
            # Amplify current activation
            # Requires capturing current output - return zero vector
            return torch.zeros(self.d_out, device=self.device)

        elif method == 'zero':
            # Zero vector (for testing)
            return torch.zeros(self.d_out, device=self.device)

        else:
            raise ValueError(f"Unknown method: {method}")

    def inject_fact(
        self,
        subject: str,
        relation: str,
        object_text: str,
        tokenizer,
        verbose: bool = False
    ) -> EditResult:
        """
        High-level interface for fact injection

        Example:
            inject_fact("Paris", "capital of", "France", tokenizer)

        Args:
            subject: Entity (e.g., "Paris")
            relation: Relation (e.g., "capital of")
            object_text: Target (e.g., "France")
            tokenizer: Tokenizer
            verbose: Print debug info

        Returns:
            EditResult with injection statistics
        """
        # Construct prompt that triggers the fact
        prompt = f"{subject} is the {relation}"

        if verbose:
            print(f"Injecting: {prompt} → {object_text}")

        try:
            # Capture activation (the "trigger")
            k_star = self.capture_activation(prompt, tokenizer)

            # Compute target (the "answer")
            v_star = self.compute_steering_vector(object_text, tokenizer)

            # Apply RLS update
            update_norm = self.inject_association(k_star, v_star, verbose=verbose)

            return EditResult(
                prompt=prompt,
                target=object_text,
                update_norm=update_norm,
                num_updates=self.num_updates,
                success=True
            )

        except Exception as e:
            if verbose:
                print(f"Error during injection: {e}")

            return EditResult(
                prompt=prompt,
                target=object_text,
                update_norm=0.0,
                num_updates=self.num_updates,
                success=False
            )

    def reset_covariance(self, regularization: Optional[float] = None):
        """Reset C^{-1} to identity (fresh start)"""
        if regularization is None:
            regularization = 1.0 / self.C_inv[0, 0].item()

        self.C_inv = torch.eye(
            self.d_in,
            device=self.device,
            dtype=torch.float32
        ) / regularization

        print(f"Covariance reset with λ={regularization:.2e}")

    def get_statistics(self) -> Dict:
        """Get editor statistics"""
        # Compute condition number of C
        try:
            C = torch.linalg.inv(self.C_inv)
            cond_number = torch.linalg.cond(C).item()
        except:
            cond_number = float('inf')

        return {
            'num_updates': self.num_updates,
            'mean_update_norm': np.mean(self.update_norms) if self.update_norms else 0.0,
            'total_update_norm': sum(self.update_norms),
            'covariance_condition': cond_number,
            'forgetting_factor': self.forgetting_factor
        }


class NullSpaceProtectedEditor(RecursiveLeastSquaresEditor):
    """
    RLS editor with null space projection

    Projects updates onto null space of "general English" to preserve
    core language capabilities while editing facts
    """

    def __init__(self, model, layer_name, **kwargs):
        super().__init__(model, layer_name, **kwargs)

        # Computed general English subspace
        self.general_subspace = None
        self.subspace_rank = 0

    def compute_general_subspace(
        self,
        general_prompts: List[str],
        tokenizer,
        rank: int = 100
    ):
        """
        Compute SVD of general English activations

        Args:
            general_prompts: List of generic English sentences
            tokenizer: Tokenizer
            rank: Number of principal components to keep
        """
        print(f"Computing general subspace from {len(general_prompts)} prompts...")

        activations = []

        for prompt in general_prompts:
            try:
                act = self.capture_activation(prompt, tokenizer)
                activations.append(act.cpu())
            except Exception as e:
                print(f"  Skipping prompt due to error: {e}")
                continue

        if len(activations) < rank:
            print(f"  Warning: Only {len(activations)} activations, less than rank={rank}")
            rank = len(activations)

        # Stack into matrix (num_prompts, d_in)
        A = torch.stack(activations)

        # SVD: A = U S V^T
        U, S, Vt = torch.linalg.svd(A.T, full_matrices=False)

        # Keep top 'rank' components
        self.general_subspace = U[:, :rank].to(self.device)  # (d_in, rank)
        self.subspace_rank = rank

        variance_explained = (S[:rank].sum() / S.sum() * 100).item()

        print(f"  General subspace: {rank} components")
        print(f"  Variance explained: {variance_explained:.1f}%")
        print(f"  Subspace norm: {self.general_subspace.norm().item():.2f}")

    def inject_association(self, k_star, v_star, verbose=False):
        """
        RLS update with null space projection

        Projects ΔW onto null space of general English before applying
        """
        # Standard RLS update (compute delta but don't apply yet)
        k_star = k_star.to(self.device).float()
        v_star = v_star.to(self.device).float()

        if k_star.dim() == 1:
            k_star = k_star.unsqueeze(1)
        if v_star.dim() == 1:
            v_star = v_star.unsqueeze(1)

        C_inv_k = self.C_inv @ k_star
        denominator = 1.0 + (k_star.T @ C_inv_k).item()
        kalman_gain = C_inv_k / denominator

        # Update C^{-1}
        if self.forgetting_factor < 1.0:
            self.C_inv = self.C_inv / self.forgetting_factor
        self.C_inv -= (kalman_gain @ C_inv_k.T) / self.forgetting_factor

        # Compute ΔW
        current_W = self.layer.weight.data
        current_output = current_W @ k_star
        error = v_star - current_output
        delta_W = error @ kalman_gain.T

        original_norm = delta_W.norm().item()

        # --- NULL SPACE PROJECTION ---
        if self.general_subspace is not None:
            # Project onto null space of V_general
            # Null space projector: P = I - V V^T

            V = self.general_subspace  # (d_in, rank)
            projector = torch.eye(self.d_in, device=self.device) - V @ V.T

            # Project ΔW: ΔW_safe = ΔW @ P
            delta_W = delta_W @ projector

            protected_norm = delta_W.norm().item()

            if verbose:
                print(f"  Null space projection:")
                print(f"    Original norm: {original_norm:.6f}")
                print(f"    Protected norm: {protected_norm:.6f}")
                print(f"    Reduction: {(1 - protected_norm/original_norm)*100:.1f}%")

        # Apply update
        self.layer.weight.data += delta_W

        update_norm = delta_W.norm().item()
        self.update_norms.append(update_norm)
        self.num_updates += 1

        return update_norm


def get_general_english_prompts() -> List[str]:
    """Get diverse general English prompts for subspace computation"""
    return [
        "The cat sat on the mat.",
        "She went to the store yesterday.",
        "It was a beautiful sunny day.",
        "The quick brown fox jumps over the lazy dog.",
        "He enjoys reading books in the library.",
        "They decided to go for a walk.",
        "The movie was very entertaining.",
        "I like to drink coffee in the morning.",
        "The children played in the park.",
        "She wrote a letter to her friend.",
        "The car stopped at the red light.",
        "He cooked dinner for his family.",
        "The dog barked at the mailman.",
        "They traveled to Europe last summer.",
        "The teacher explained the lesson clearly.",
        "She loves listening to classical music.",
        "The flowers bloomed in spring.",
        "He works in an office downtown.",
        "The baby laughed at the funny faces.",
        "They enjoyed the concert very much.",
        # Add more for better subspace estimation
    ]


if __name__ == "__main__":
    print("RLS Weight Editor - Demo")
    print("="*60)
    print("\nThis module provides tools for one-shot knowledge injection")
    print("using Recursive Least Squares with Sherman-Morrison updates.")
    print("\nSee test_rls.py for usage examples.")
