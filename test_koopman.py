#!/usr/bin/env python3
"""
Test Koopman DMD on transformer hidden states

This script loads collected hidden states and tests DMD prediction accuracy.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse

from koopman_dmd import KoopmanDMD, KoopmanVisualizer


class KoopmanEvaluator:
    """Evaluate DMD prediction on transformer trajectories"""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing hidden state data
        """
        self.data_dir = Path(data_dir)

        # Load metadata
        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        print(f"Loaded data from: {data_dir}")
        print(f"  Model: {self.metadata['model_name']}")
        print(f"  Layers: {self.metadata['num_layers']}")
        print(f"  Prompts: {self.metadata['num_prompts']}")

    def load_trajectory(self, idx: int) -> Tuple[str, torch.Tensor]:
        """Load single trajectory"""
        trajectory_file = self.data_dir / f"trajectory_{idx:04d}.pt"
        data = torch.load(trajectory_file)
        return data['prompt'], data['trajectory']

    def test_single_prediction(
        self,
        trajectory_idx: int,
        observe_layers: int = 15,
        target_layer: int = 32,
        rank: int = 10
    ) -> Dict:
        """
        Test DMD prediction on single trajectory

        Args:
            trajectory_idx: Index of trajectory to test
            observe_layers: Number of layers to observe
            target_layer: Layer to predict
            rank: DMD rank

        Returns:
            Dictionary with results
        """
        prompt, full_trajectory = self.load_trajectory(trajectory_idx)

        # Observed portion
        observed = full_trajectory[:observe_layers, :]

        # Fit DMD
        dmd = KoopmanDMD(rank=rank)
        dmd.fit(observed)

        # Predict target layer
        predicted = dmd.predict(target_layer=target_layer, current_layer=observe_layers)

        # Ground truth
        true_state = full_trajectory[target_layer - 1, :]  # 0-indexed

        # Compute metrics
        cosine_sim = F.cosine_similarity(
            predicted.unsqueeze(0),
            true_state.unsqueeze(0)
        ).item()

        rel_error = (torch.norm(predicted - true_state) / torch.norm(true_state)).item()

        # Analyze spectrum
        spectrum, mags, phases = dmd.analyze_spectrum()
        participation = dmd.mode_participation()

        return {
            'prompt': prompt,
            'cosine_similarity': cosine_sim,
            'relative_error': rel_error,
            'spectrum': spectrum,
            'eigenvalue_magnitudes': mags.cpu().numpy(),
            'eigenvalue_phases': phases.cpu().numpy(),
            'mode_participation': participation.cpu().numpy(),
            'dmd': dmd
        }

    def sweep_observation_windows(
        self,
        trajectory_idx: int,
        observe_range: List[int] = [5, 10, 15, 20, 25],
        target_range: List[int] = None,
        rank: int = 10
    ) -> np.ndarray:
        """
        Test prediction accuracy for different observation/target combinations

        Args:
            trajectory_idx: Trajectory to test
            observe_range: List of observation window sizes
            target_range: List of target layers (None = max available)
            rank: DMD rank

        Returns:
            Accuracy matrix (observe_range × target_range)
        """
        prompt, full_trajectory = self.load_trajectory(trajectory_idx)
        max_layers = full_trajectory.shape[0]

        if target_range is None:
            target_range = list(range(10, max_layers + 1, 2))

        accuracy_matrix = np.zeros((len(observe_range), len(target_range)))

        for i, obs in enumerate(observe_range):
            for j, tgt in enumerate(target_range):
                if tgt <= obs:
                    continue  # Can't predict backwards

                try:
                    # Fit DMD
                    observed = full_trajectory[:obs, :]
                    dmd = KoopmanDMD(rank=rank)
                    dmd.fit(observed)

                    # Predict
                    predicted = dmd.predict(target_layer=tgt, current_layer=obs)
                    true_state = full_trajectory[tgt - 1, :]

                    # Cosine similarity
                    cosine_sim = F.cosine_similarity(
                        predicted.unsqueeze(0),
                        true_state.unsqueeze(0)
                    ).item()

                    accuracy_matrix[i, j] = cosine_sim

                except Exception as e:
                    print(f"Error at obs={obs}, tgt={tgt}: {e}")
                    accuracy_matrix[i, j] = 0.0

        return accuracy_matrix, observe_range, target_range

    def analyze_all_trajectories(
        self,
        observe_layers: int = 15,
        target_layer: int = 32,
        rank: int = 10
    ) -> Dict:
        """
        Analyze all trajectories in dataset

        Returns:
            Dictionary with aggregated statistics
        """
        num_trajectories = self.metadata['num_prompts']

        results = []

        for i in tqdm(range(num_trajectories), desc="Analyzing trajectories"):
            try:
                result = self.test_single_prediction(
                    trajectory_idx=i,
                    observe_layers=observe_layers,
                    target_layer=target_layer,
                    rank=rank
                )
                results.append(result)
            except Exception as e:
                print(f"Error on trajectory {i}: {e}")
                continue

        # Aggregate statistics
        cosine_sims = [r['cosine_similarity'] for r in results]
        rel_errors = [r['relative_error'] for r in results]

        aggregated = {
            'num_tested': len(results),
            'observe_layers': observe_layers,
            'target_layer': target_layer,
            'rank': rank,
            'cosine_similarity': {
                'mean': np.mean(cosine_sims),
                'std': np.std(cosine_sims),
                'min': np.min(cosine_sims),
                'max': np.max(cosine_sims),
                'median': np.median(cosine_sims),
            },
            'relative_error': {
                'mean': np.mean(rel_errors),
                'std': np.std(rel_errors),
                'min': np.min(rel_errors),
                'max': np.max(rel_errors),
                'median': np.median(rel_errors),
            },
            'individual_results': results
        }

        return aggregated

    def find_optimal_rank(
        self,
        trajectory_idx: int,
        observe_layers: int = 15,
        target_layer: int = 32,
        rank_range: List[int] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Find optimal DMD rank by testing range

        Returns:
            (ranks, accuracies)
        """
        if rank_range is None:
            rank_range = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]

        prompt, full_trajectory = self.load_trajectory(trajectory_idx)
        observed = full_trajectory[:observe_layers, :]
        true_state = full_trajectory[target_layer - 1, :]

        accuracies = []

        for rank in rank_range:
            try:
                dmd = KoopmanDMD(rank=rank)
                dmd.fit(observed)

                predicted = dmd.predict(target_layer=target_layer,
                                      current_layer=observe_layers)

                cosine_sim = F.cosine_similarity(
                    predicted.unsqueeze(0),
                    true_state.unsqueeze(0)
                ).item()

                accuracies.append(cosine_sim)

            except Exception as e:
                print(f"Error at rank={rank}: {e}")
                accuracies.append(0.0)

        return rank_range, accuracies


def main():
    parser = argparse.ArgumentParser(description="Test Koopman DMD on transformers")
    parser.add_argument("--data-dir", type=str, default="hidden_states_data",
                       help="Directory with hidden state data")
    parser.add_argument("--observe-layers", type=int, default=15,
                       help="Number of layers to observe")
    parser.add_argument("--target-layer", type=int, default=32,
                       help="Target layer to predict")
    parser.add_argument("--rank", type=int, default=10,
                       help="DMD rank")
    parser.add_argument("--test-idx", type=int, default=0,
                       help="Trajectory index to test")
    parser.add_argument("--mode", type=str, default="single",
                       choices=["single", "all", "sweep", "rank-search"],
                       help="Testing mode")
    parser.add_argument("--output-dir", type=str, default="koopman_results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize evaluator
    evaluator = KoopmanEvaluator(args.data_dir)

    if args.mode == "single":
        print(f"\n{'='*60}")
        print(f"SINGLE TRAJECTORY TEST")
        print(f"{'='*60}\n")

        result = evaluator.test_single_prediction(
            trajectory_idx=args.test_idx,
            observe_layers=args.observe_layers,
            target_layer=args.target_layer,
            rank=args.rank
        )

        print(f"Prompt: {result['prompt']}")
        print(f"\nPrediction ({args.observe_layers} → {args.target_layer}):")
        print(f"  Cosine similarity: {result['cosine_similarity']:.6f}")
        print(f"  Relative error: {result['relative_error']:.6f}")

        print(f"\nSpectrum analysis:")
        for key, value in result['spectrum'].items():
            print(f"  {key}: {value}")

        # Visualize eigenvalues
        fig1 = KoopmanVisualizer.plot_eigenvalue_spectrum(
            result['dmd'].Lambda,
            save_path=output_dir / f"eigenvalues_{args.test_idx}.png"
        )
        fig1.suptitle(f"Koopman Spectrum\n{result['prompt'][:60]}...",
                     fontsize=12, fontweight='bold')

        # Visualize mode participation
        fig2 = KoopmanVisualizer.plot_mode_participation(
            torch.tensor(result['mode_participation']),
            save_path=output_dir / f"modes_{args.test_idx}.png"
        )

        plt.show()

    elif args.mode == "all":
        print(f"\n{'='*60}")
        print(f"ALL TRAJECTORIES ANALYSIS")
        print(f"{'='*60}\n")

        results = evaluator.analyze_all_trajectories(
            observe_layers=args.observe_layers,
            target_layer=args.target_layer,
            rank=args.rank
        )

        print(f"\nTested: {results['num_tested']} trajectories")
        print(f"Configuration: {args.observe_layers} → {args.target_layer}, rank={args.rank}")

        print(f"\nCosine Similarity:")
        for key, value in results['cosine_similarity'].items():
            print(f"  {key}: {value:.6f}")

        print(f"\nRelative Error:")
        for key, value in results['relative_error'].items():
            print(f"  {key}: {value:.6f}")

        # Save results
        with open(output_dir / "all_trajectories_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON
            json_results = results.copy()
            json_results['individual_results'] = [
                {k: v.tolist() if isinstance(v, np.ndarray) else v
                 for k, v in r.items() if k != 'dmd'}
                for r in results['individual_results']
            ]
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to: {output_dir}/all_trajectories_results.json")

    elif args.mode == "sweep":
        print(f"\n{'='*60}")
        print(f"OBSERVATION WINDOW SWEEP")
        print(f"{'='*60}\n")

        accuracy_matrix, observe_range, target_range = evaluator.sweep_observation_windows(
            trajectory_idx=args.test_idx,
            rank=args.rank
        )

        print(f"Tested observation windows: {observe_range}")
        print(f"Tested target layers: {target_range[:5]}...{target_range[-5:]}")

        # Find best combination
        best_idx = np.unravel_index(accuracy_matrix.argmax(), accuracy_matrix.shape)
        best_obs = observe_range[best_idx[0]]
        best_tgt = target_range[best_idx[1]]
        best_acc = accuracy_matrix[best_idx]

        print(f"\nBest configuration:")
        print(f"  Observe: {best_obs} layers")
        print(f"  Target: {best_tgt} layers")
        print(f"  Accuracy: {best_acc:.6f}")

        # Visualize
        fig = KoopmanVisualizer.plot_prediction_accuracy(
            observe_range,
            target_range,
            accuracy_matrix,
            save_path=output_dir / f"accuracy_heatmap_{args.test_idx}.png"
        )
        plt.show()

    elif args.mode == "rank-search":
        print(f"\n{'='*60}")
        print(f"RANK OPTIMIZATION")
        print(f"{'='*60}\n")

        ranks, accuracies = evaluator.find_optimal_rank(
            trajectory_idx=args.test_idx,
            observe_layers=args.observe_layers,
            target_layer=args.target_layer
        )

        print(f"Tested ranks: {ranks}")

        # Find optimal
        best_idx = np.argmax(accuracies)
        best_rank = ranks[best_idx]
        best_acc = accuracies[best_idx]

        print(f"\nOptimal rank: {best_rank}")
        print(f"Accuracy: {best_acc:.6f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ranks, accuracies, 'bo-', linewidth=2, markersize=8)
        ax.axvline(best_rank, color='r', linestyle='--', alpha=0.5,
                  label=f'Optimal: rank={best_rank}')
        ax.set_xlabel('DMD Rank', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Prediction Accuracy vs. DMD Rank', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.savefig(output_dir / f"rank_optimization_{args.test_idx}.png",
                   dpi=300, bbox_inches='tight')
        plt.show()

    print(f"\n✅ Analysis complete! Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
