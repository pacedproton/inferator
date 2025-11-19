#!/usr/bin/env python3
"""
Kalman Filtering for Hidden State Tracking

Smooth hidden state evolution using Kalman filter to reduce noise
and improve stability.

Mathematical foundation:
- Kalman filtering (optimal state estimation)
- Linear dynamical systems
- Gaussian state-space models
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class KalmanState:
    """Kalman filter state"""
    mean: torch.Tensor  # State estimate
    covariance: torch.Tensor  # Uncertainty
    innovation: float  # Prediction error
    gain: torch.Tensor  # Kalman gain


class KalmanFilter:
    """
    Kalman filter for hidden state tracking

    State-space model:
    x_t = F x_{t-1} + w_t    (process model)
    z_t = H x_t + v_t        (observation model)

    where:
    - x_t = true state
    - z_t = observation
    - F = state transition matrix
    - H = observation matrix
    - w_t ~ N(0, Q) = process noise
    - v_t ~ N(0, R) = observation noise

    Kalman filter computes optimal estimate x̂_t given observations z_{1:t}
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
        init_variance: float = 1.0
    ):
        """
        Args:
            state_dim: Dimension of state
            obs_dim: Dimension of observations
            process_noise: Process noise variance (Q)
            observation_noise: Observation noise variance (R)
            init_variance: Initial state variance
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # State estimate
        self.x_hat = None  # Mean
        self.P = None      # Covariance

        # Model matrices (will be learned or set)
        self.F = torch.eye(state_dim)  # Default: identity (random walk)
        self.H = torch.eye(obs_dim, state_dim)  # Default: identity observation

        # Noise covariances
        self.Q = torch.eye(state_dim) * process_noise
        self.R = torch.eye(obs_dim) * observation_noise

        # Initial covariance
        self.init_variance = init_variance

        print(f"KalmanFilter initialized:")
        print(f"  State dim: {state_dim}")
        print(f"  Obs dim: {obs_dim}")
        print(f"  Process noise: {process_noise}")
        print(f"  Observation noise: {observation_noise}")

    def initialize(self, initial_state: Optional[torch.Tensor] = None):
        """
        Initialize filter state

        Args:
            initial_state: Initial state estimate (if None, use zero)
        """
        if initial_state is None:
            self.x_hat = torch.zeros(self.state_dim)
        else:
            self.x_hat = initial_state.clone()

        self.P = torch.eye(self.state_dim) * self.init_variance

    def predict(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction step: x̄_t = F x̂_{t-1}
                        P̄_t = F P_{t-1} F^T + Q

        Returns:
            (predicted_state, predicted_covariance)
        """
        if self.x_hat is None:
            self.initialize()

        # Predict state
        x_bar = self.F @ self.x_hat

        # Predict covariance
        P_bar = self.F @ self.P @ self.F.T + self.Q

        return x_bar, P_bar

    def update(
        self,
        observation: torch.Tensor,
        x_bar: Optional[torch.Tensor] = None,
        P_bar: Optional[torch.Tensor] = None
    ) -> KalmanState:
        """
        Update step given observation

        Kalman gain: K_t = P̄_t H^T (H P̄_t H^T + R)^{-1}
        State update: x̂_t = x̄_t + K_t (z_t - H x̄_t)
        Covariance update: P_t = (I - K_t H) P̄_t

        Args:
            observation: z_t
            x_bar: Predicted state (if None, run predict first)
            P_bar: Predicted covariance

        Returns:
            Updated Kalman state
        """
        # Predict if not provided
        if x_bar is None or P_bar is None:
            x_bar, P_bar = self.predict()

        # Innovation: y_t = z_t - H x̄_t
        innovation = observation - self.H @ x_bar

        # Innovation covariance: S_t = H P̄_t H^T + R
        S = self.H @ P_bar @ self.H.T + self.R

        # Kalman gain: K_t = P̄_t H^T S_t^{-1}
        K = P_bar @ self.H.T @ torch.inverse(S)

        # State update
        self.x_hat = x_bar + K @ innovation

        # Covariance update (Joseph form for numerical stability)
        I_KH = torch.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ P_bar @ I_KH.T + K @ self.R @ K.T

        # Compute innovation norm
        innovation_norm = torch.norm(innovation).item()

        return KalmanState(
            mean=self.x_hat.clone(),
            covariance=self.P.clone(),
            innovation=innovation_norm,
            gain=K.clone()
        )

    def filter_sequence(
        self,
        observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter entire sequence of observations

        Args:
            observations: (seq_len, obs_dim)

        Returns:
            (filtered_states, filtered_covariances)
            - filtered_states: (seq_len, state_dim)
            - filtered_covariances: (seq_len, state_dim, state_dim)
        """
        seq_len = observations.shape[0]

        states = []
        covariances = []

        self.initialize(observations[0])

        for t in range(seq_len):
            # Predict
            x_bar, P_bar = self.predict()

            # Update
            kalman_state = self.update(observations[t], x_bar, P_bar)

            states.append(kalman_state.mean)
            covariances.append(kalman_state.covariance)

        return torch.stack(states), torch.stack(covariances)

    def smooth_sequence(
        self,
        observations: torch.Tensor
    ) -> torch.Tensor:
        """
        Rauch-Tung-Striebel (RTS) smoother

        Forward-backward pass for smoothing

        Args:
            observations: (seq_len, obs_dim)

        Returns:
            Smoothed states (seq_len, state_dim)
        """
        seq_len = observations.shape[0]

        # Forward pass (filtering)
        filtered_states, filtered_covs = self.filter_sequence(observations)

        # Backward pass (smoothing)
        smoothed_states = [filtered_states[-1]]
        smoothed_covs = [filtered_covs[-1]]

        for t in range(seq_len - 2, -1, -1):
            # Smoother gain
            P_pred = self.F @ filtered_covs[t] @ self.F.T + self.Q
            G = filtered_covs[t] @ self.F.T @ torch.inverse(P_pred)

            # Smooth state
            x_smooth = filtered_states[t] + G @ (
                smoothed_states[0] - self.F @ filtered_states[t]
            )

            # Smooth covariance
            P_smooth = filtered_covs[t] + G @ (
                smoothed_covs[0] - P_pred
            ) @ G.T

            smoothed_states.insert(0, x_smooth)
            smoothed_covs.insert(0, P_smooth)

        return torch.stack(smoothed_states)


class KalmanTransformer(nn.Module):
    """
    Apply Kalman filtering to transformer hidden states

    Smooths hidden state evolution across layers
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
        use_smoothing: bool = True
    ):
        """
        Args:
            hidden_dim: Hidden state dimension
            num_layers: Number of transformer layers
            process_noise: Process noise (Q)
            observation_noise: Observation noise (R)
            use_smoothing: Use RTS smoother (requires full sequence)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_smoothing = use_smoothing

        # Create Kalman filter for each position in sequence
        # (In practice, share parameters across positions)
        self.kf = KalmanFilter(
            state_dim=hidden_dim,
            obs_dim=hidden_dim,
            process_noise=process_noise,
            observation_noise=observation_noise
        )

        # Learnable transition matrix
        self.F_net = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.F_net.weight)

        print(f"KalmanTransformer:")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Use smoothing: {use_smoothing}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Apply Kalman filtering to hidden states

        Args:
            hidden_states: (batch, num_layers, seq_len, hidden_dim)
            return_stats: Return filtering statistics

        Returns:
            (filtered_states, stats)
        """
        batch, num_layers, seq_len, hidden_dim = hidden_states.shape

        # Update transition matrix
        self.kf.F = self.F_net.weight.data

        filtered = torch.zeros_like(hidden_states)

        # Process each batch and sequence position independently
        for b in range(batch):
            for s in range(seq_len):
                # Extract trajectory across layers
                trajectory = hidden_states[b, :, s, :]  # (num_layers, hidden_dim)

                if self.use_smoothing:
                    # RTS smoothing (forward-backward)
                    smoothed = self.kf.smooth_sequence(trajectory)
                    filtered[b, :, s, :] = smoothed
                else:
                    # Forward-only filtering
                    filt, _ = self.kf.filter_sequence(trajectory)
                    filtered[b, :, s, :] = filt

        stats = None
        if return_stats:
            # Compute noise reduction
            original_var = hidden_states.var(dim=(0, 1, 2)).mean().item()
            filtered_var = filtered.var(dim=(0, 1, 2)).mean().item()
            noise_reduction = 1.0 - (filtered_var / original_var)

            stats = {
                'original_variance': original_var,
                'filtered_variance': filtered_var,
                'noise_reduction': noise_reduction
            }

        return filtered, stats


def demo_kalman_filtering():
    """Demonstrate Kalman filtering"""
    print("Kalman Filtering Demo")
    print("=" * 60)

    # Simulate noisy observations
    true_state_dim = 2
    obs_dim = 2
    seq_len = 50

    # True trajectory (sine wave)
    t = torch.linspace(0, 4*np.pi, seq_len)
    true_states = torch.stack([torch.sin(t), torch.cos(t)], dim=1)

    # Noisy observations
    observations = true_states + torch.randn_like(true_states) * 0.3

    print(f"\nTrue state dim: {true_state_dim}")
    print(f"Sequence length: {seq_len}")
    print(f"Observation noise std: 0.3")

    # Create Kalman filter
    kf = KalmanFilter(
        state_dim=true_state_dim,
        obs_dim=obs_dim,
        process_noise=0.01,
        observation_noise=0.1
    )

    # Filter
    filtered_states, _ = kf.filter_sequence(observations)

    # Smooth
    kf2 = KalmanFilter(
        state_dim=true_state_dim,
        obs_dim=obs_dim,
        process_noise=0.01,
        observation_noise=0.1
    )
    smoothed_states = kf2.smooth_sequence(observations)

    # Compute errors
    obs_error = torch.norm(observations - true_states, dim=1).mean()
    filtered_error = torch.norm(filtered_states - true_states, dim=1).mean()
    smoothed_error = torch.norm(smoothed_states - true_states, dim=1).mean()

    print(f"\nMean errors:")
    print(f"  Observations: {obs_error:.4f}")
    print(f"  Filtered: {filtered_error:.4f}")
    print(f"  Smoothed: {smoothed_error:.4f}")

    improvement_filtered = (obs_error - filtered_error) / obs_error * 100
    improvement_smoothed = (obs_error - smoothed_error) / obs_error * 100

    print(f"\nImprovements:")
    print(f"  Filtering: {improvement_filtered:.1f}%")
    print(f"  Smoothing: {improvement_smoothed:.1f}%")

    print(f"\n✅ Kalman filtering reduces noise!")


if __name__ == "__main__":
    print("Kalman Filtering for Hidden State Tracking")
    print("=" * 60)
    print("\nOptimal state estimation for noisy observations\n")

    demo_kalman_filtering()

    print("\n" + "=" * 60)
    print("Key Insights:")
    print("  - Kalman filter is optimal for linear Gaussian systems")
    print("  - Combines predictions with observations")
    print("  - RTS smoother uses future information (offline)")
    print("  - Reduces noise and improves stability")
