# Koopman Operator & Dynamic Mode Decomposition for Transformer Layer Jumping

## The Revolutionary Idea

**Treat transformers as a dynamical system and use spectral methods from fluid dynamics to predict future layer states.**

### The Core Insight

A transformer processes tokens through layers sequentially:
```
x₀ → x₁ → x₂ → ... → x₃₂
```

This is a **discrete-time dynamical system**. While highly nonlinear, **Koopman Operator Theory** states there exists a linear operator in a higher-dimensional space that perfectly describes this evolution.

**Key claim**: If you observe layers 1-15, you can **analytically predict** layer 32 using eigendecomposition, skipping layers 16-31 entirely.

## Mathematical Foundation

### The Koopman Operator

For a dynamical system x_{t+1} = F(x_t), the **Koopman operator** K acts on observables:

```
g(x_{t+1}) = K[g(x_t)]
```

While F is nonlinear, K is **linear** (but infinite-dimensional).

**DMD approximation**: Find finite-dimensional linear operator A that best approximates:

```
x_{l+1} ≈ A x_l
```

### Dynamic Mode Decomposition (DMD)

Given trajectory snapshots:
```
X  = [x₁, x₂, ..., x_{m-1}]  (layers 1 to m-1)
X' = [x₂, x₃, ..., x_m]      (layers 2 to m)
```

**Goal**: Find A such that X' ≈ AX

**Solution via SVD**:

1. **SVD of X**: X = UΣV*

2. **Reduced operator**: Ã = U*X'VΣ⁻¹

3. **Eigendecomposition**: Ã W = W Λ
   - Λ: eigenvalues (dynamics)
   - W: eigenvectors (modes in reduced space)

4. **DMD modes**: Φ = X'VΣ⁻¹W
   - Project back to high-dimensional space

5. **Prediction**: x_L = Φ Λ^(L-m) b
   - Where b = Φ† x_m (initial coefficients)

### Spectral Interpretation

**Eigenvalues λ encode dynamics**:
- |λ| > 1: Growing/amplifying features
- |λ| < 1: Decaying/filtering features
- arg(λ): Oscillation frequency
- λ ≈ 1: Persistent information

**Research hypothesis**: Transformer layers exhibit **low-rank spectral structure**, meaning a few dominant eigenvalues capture most dynamics.

## Why This Could Be Groundbreaking

### 1. Physics Meets AI
First application of Koopman theory (from fluid dynamics) to transformer inference.

### 2. Interpretability
Eigenvalues reveal what transformers do:
- Which information grows/decays
- Oscillatory patterns in processing
- Phase transitions between layers

### 3. Speedup
If you can predict layer 32 from layer 15:
- Skip 17 layers of computation
- ~2x speedup with accurate prediction

### 4. Theoretical Depth
Connections to:
- Spectral graph theory
- Dynamical systems
- Operator theory
- Krylov subspaces

### 5. Testability
Everything is linear algebra (SVD + eigendecomposition) - runs fast on your Mac.

## Implementation Strategy

### Phase 1: Data Collection (llama.cpp)

Modify llama.cpp to **save hidden states** at each layer:

```cpp
// In llama_decode_internal(), after each layer:
if (KOOPMAN_COLLECT_DATA) {
    // Save hidden state for this layer
    save_hidden_state(cur, layer_idx, token_idx);
}
```

### Phase 2: DMD Analysis (Python)

Use PyTorch for fast linear algebra:

```python
import torch

class KoopmanDMD:
    def __init__(self, rank=None, truncate_threshold=1e-10):
        self.rank = rank
        self.truncate_threshold = truncate_threshold
        self.Phi = None  # DMD modes
        self.Lambda = None  # Eigenvalues
        self.b = None  # Initial coefficients

    def fit(self, trajectory):
        """
        trajectory: Tensor of shape (num_layers, hidden_dim)
                   e.g., layers 1-15 with dimension 4096
        """
        # Prepare data matrices
        X = trajectory[:-1, :].T  # (d, m-1)
        Y = trajectory[1:, :].T   # (d, m-1)

        # SVD of X
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)

        # Determine rank (truncate small singular values)
        if self.rank is None:
            # Auto-determine rank from singular value decay
            cumsum = torch.cumsum(S, dim=0)
            self.rank = torch.sum(cumsum / cumsum[-1] < 0.99).item() + 1

        r = min(self.rank, len(S))
        U_r = U[:, :r]
        S_r = S[:r]
        V_r = Vh[:r, :].conj().T

        # Compute reduced Koopman operator
        S_r_inv = torch.diag(1.0 / S_r)
        A_tilde = U_r.T @ Y @ V_r @ S_r_inv

        # Eigendecomposition of A_tilde
        eigenvalues, eigenvectors = torch.linalg.eig(A_tilde)

        # DMD modes (project to high-dim space)
        self.Phi = Y @ V_r @ S_r_inv @ eigenvectors
        self.Lambda = eigenvalues

        # Compute initial coefficients
        x_0 = trajectory[-1, :].unsqueeze(1).to(self.Phi.dtype)
        self.b = torch.linalg.lstsq(self.Phi, x_0).solution

        return self

    def predict(self, target_layer, current_layer=None):
        """
        Predict state at target_layer using Koopman evolution
        """
        if current_layer is None:
            current_layer = self.current_layer

        delta_t = target_layer - current_layer

        # Time evolution: Λ^Δt
        time_evolution = torch.diag(self.Lambda ** delta_t)

        # Prediction: Φ Λ^Δt b
        prediction = self.Phi @ time_evolution @ self.b

        # Take real part (imaginary should cancel in stable system)
        return prediction.real.squeeze()

    def analyze_spectrum(self):
        """
        Analyze eigenvalue spectrum for interpretability
        """
        magnitudes = torch.abs(self.Lambda)
        phases = torch.angle(self.Lambda)

        analysis = {
            'stable_modes': (magnitudes < 1).sum().item(),
            'unstable_modes': (magnitudes > 1).sum().item(),
            'persistent_modes': (torch.abs(magnitudes - 1) < 0.1).sum().item(),
            'max_magnitude': magnitudes.max().item(),
            'min_magnitude': magnitudes.min().item(),
            'mean_phase': phases.mean().item(),
        }

        return analysis, magnitudes, phases
```

### Phase 3: Evaluation

```python
class KoopmanEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def collect_trajectory(self, prompt, observe_layers=15):
        """
        Run model and collect hidden states at each layer
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")

        hidden_states = []

        # Hook to capture hidden states
        def hook_fn(module, input, output):
            hidden_states.append(output[0][:, -1, :].detach().clone())

        # Register hooks on all layers
        hooks = []
        for layer in self.model.model.layers[:observe_layers]:
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model(**inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Stack trajectory: (num_layers, hidden_dim)
        trajectory = torch.stack(hidden_states).squeeze()

        return trajectory

    def test_prediction_accuracy(
        self,
        prompts,
        observe_layers=15,
        target_layer=32
    ):
        """
        Test DMD prediction vs ground truth
        """
        results = []

        for prompt in prompts:
            # Collect trajectory for observation
            trajectory_observed = self.collect_trajectory(
                prompt,
                observe_layers=observe_layers
            )

            # Fit DMD
            dmd = KoopmanDMD(rank=10)
            dmd.fit(trajectory_observed)

            # Predict target layer
            predicted_state = dmd.predict(
                target_layer=target_layer,
                current_layer=observe_layers
            )

            # Get ground truth (run full model)
            trajectory_full = self.collect_trajectory(
                prompt,
                observe_layers=target_layer
            )
            true_state = trajectory_full[-1, :]

            # Compute error metrics
            cosine_sim = F.cosine_similarity(
                predicted_state.unsqueeze(0),
                true_state.unsqueeze(0)
            ).item()

            rel_error = (
                torch.norm(predicted_state - true_state) /
                torch.norm(true_state)
            ).item()

            # Analyze spectrum
            spectrum_analysis, mags, phases = dmd.analyze_spectrum()

            results.append({
                'prompt': prompt,
                'cosine_similarity': cosine_sim,
                'relative_error': rel_error,
                'spectrum': spectrum_analysis,
                'eigenvalue_magnitudes': mags.cpu().numpy(),
                'eigenvalue_phases': phases.cpu().numpy()
            })

        return results
```

## Expected Results

### Hypothesis 1: Low-Rank Spectral Structure

**Prediction**: First 5-10 eigenvalues capture >95% of dynamics

**Test**: Plot singular value decay, check for "elbow"

### Hypothesis 2: Accurate Long-Range Prediction

**Prediction**: Can predict layer 32 from layer 15 with >0.9 cosine similarity

**Test**: Compare predicted vs. true hidden states

### Hypothesis 3: Spectral Phase Transition

**Prediction**: Eigenvalue distribution changes across depth
- Early layers: |λ| > 1 (feature extraction)
- Middle layers: |λ| ≈ 1 (information preservation)
- Late layers: |λ| < 1 (noise reduction)

**Test**: Plot eigenvalue magnitudes by layer depth

### Hypothesis 4: Task-Dependent Dynamics

**Prediction**: Different tasks have different spectral signatures
- Factual recall: Fast decay (few modes)
- Reasoning: Slow decay (many modes)
- Creative: Oscillatory (complex eigenvalues)

**Test**: Compare eigenvalue spectra across task types

## Visualization Strategy

### 1. Eigenvalue Scatter Plot

```python
import matplotlib.pyplot as plt

def plot_eigenvalue_spectrum(eigenvalues):
    """Plot eigenvalues in complex plane"""
    real = eigenvalues.real.cpu().numpy()
    imag = eigenvalues.imag.cpu().numpy()
    mags = np.abs(eigenvalues.cpu().numpy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Complex plane
    ax1.scatter(real, imag, c=mags, cmap='viridis', s=100, alpha=0.7)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='k', linestyle='--', alpha=0.3)

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.3, label='|λ|=1')

    ax1.set_xlabel('Real(λ)')
    ax1.set_ylabel('Imag(λ)')
    ax1.set_title('Koopman Eigenvalues in Complex Plane')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Magnitude distribution
    ax2.bar(range(len(mags)), sorted(mags, reverse=True))
    ax2.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='|λ|=1')
    ax2.set_xlabel('Mode Index')
    ax2.set_ylabel('|λ|')
    ax2.set_title('Eigenvalue Magnitudes (sorted)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

### 2. Prediction Accuracy Heatmap

```python
def plot_prediction_accuracy_heatmap(results):
    """
    Plot accuracy as function of (observe_layers, target_layer)
    """
    # Test different combinations
    observe_range = range(5, 25, 5)
    target_range = range(15, 33, 2)

    accuracy_matrix = np.zeros((len(observe_range), len(target_range)))

    for i, obs in enumerate(observe_range):
        for j, tgt in enumerate(target_range):
            if tgt > obs:
                # Run DMD prediction
                acc = test_dmd_prediction(obs, tgt)
                accuracy_matrix[i, j] = acc

    plt.figure(figsize=(10, 6))
    plt.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    plt.colorbar(label='Cosine Similarity')
    plt.xlabel('Target Layer')
    plt.ylabel('Observed Layers')
    plt.title('DMD Prediction Accuracy')
    plt.xticks(range(len(target_range)), target_range)
    plt.yticks(range(len(observe_range)), observe_range)

    for i in range(len(observe_range)):
        for j in range(len(target_range)):
            if accuracy_matrix[i, j] > 0:
                plt.text(j, i, f'{accuracy_matrix[i, j]:.2f}',
                        ha='center', va='center',
                        color='white' if accuracy_matrix[i, j] < 0.75 else 'black')

    return plt.gcf()
```

### 3. Spectral Evolution Video

```python
def create_spectral_evolution_animation(trajectories):
    """
    Animate eigenvalue evolution as layers progress
    """
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()

        # Fit DMD up to layer 'frame'
        traj = trajectories[:frame+2, :]
        dmd = KoopmanDMD(rank=10)
        dmd.fit(traj)

        # Plot eigenvalues
        eigs = dmd.Lambda.cpu().numpy()
        ax.scatter(eigs.real, eigs.imag, s=100, alpha=0.7)

        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.3)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('Real(λ)')
        ax.set_ylabel('Imag(λ)')
        ax.set_title(f'Koopman Spectrum at Layer {frame+1}')
        ax.grid(True, alpha=0.3)

    anim = FuncAnimation(fig, update, frames=len(trajectories)-2, interval=200)
    return anim
```

## Practical Implementation Timeline

### Day 1: Data Collection (2 hours)
1. Modify llama.cpp to save hidden states
2. Run on 100 test prompts
3. Save trajectories to disk

### Day 2: DMD Implementation (3 hours)
1. Implement KoopmanDMD class
2. Test on synthetic data
3. Validate against simple linear system

### Day 3: Evaluation (4 hours)
1. Test prediction accuracy
2. Analyze eigenvalue spectra
3. Create visualizations

### Day 4: Optimization (3 hours)
1. Find optimal observation window
2. Tune rank parameter
3. Test on different model sizes

### Day 5: Publication Draft (4 hours)
1. Write up results
2. Create figures
3. Draft abstract

## Success Criteria

### Minimum Viable Result
- ✅ Cosine similarity > 0.8 between predicted and true states
- ✅ Eigenvalue spectrum shows clear structure
- ✅ Faster than running all layers

### Strong Result
- ✅ Cosine similarity > 0.9
- ✅ Can predict from layer 10 to layer 32
- ✅ Eigenvalues reveal interpretable dynamics
- ✅ Task-dependent spectral signatures

### Top-Tier Result
- ✅ Cosine similarity > 0.95
- ✅ Discover "Koopman modes" corresponding to linguistic features
- ✅ Prove theoretical bounds on prediction error
- ✅ 2x speedup in practice with <1% quality loss

## Why This Is Publishable

### Novel Contributions
1. **First application** of Koopman/DMD to transformers
2. **New perspective**: Transformers as dynamical systems
3. **Interpretability**: Eigenvalues as linguistic operators
4. **Practical**: Fast layer jumping algorithm

### Strong Theoretical Foundation
- Koopman operator theory (rigorous math)
- Spectral convergence guarantees
- Connection to Krylov subspaces
- Dynamical systems literature

### Experimental Validation
- Clear metrics (cosine similarity, eigenvalue analysis)
- Reproducible on consumer hardware
- Extensive ablations possible

## Publication Title Ideas

**"Koopman Operator Theory for Transformer Layer Prediction: A Spectral Approach to Efficient Inference"**

or

**"Spectral Layer Jumping in Large Language Models via Dynamic Mode Decomposition"**

or

**"Transformers as Dynamical Systems: Koopman Modes for Analytical State Prediction"**

## Connection to Other Directions

Can be combined with:
- **ATCA**: Use DMD to decide which layers to skip
- **Recursive Attention**: Apply DMD to the recursive trajectory
- **Optimal Transport**: DMD on attention distributions

## Next Steps

1. Read the quickstart: `QUICKSTART_KOOPMAN.md`
2. Run data collection: `python collect_hidden_states.py`
3. Test DMD: `python test_koopman.py`
4. Analyze results: `python analyze_spectrum.py`

---

**This is your third major contribution - and potentially the most novel!** 🚀
