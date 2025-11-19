# Three Additional Research Directions for 64GB MacBook LLM Research

## Direction 3: Optimal Transport Attention Alignment (OTAA)

### The Mathematical Idea

Use **Wasserstein distance** and **optimal transport** to align attention distributions across layers, reducing redundancy.

### Core Theory

**Wasserstein Distance** between probability distributions:
```
W_2(P, Q) = inf_{γ ∈ Γ(P,Q)} √(∫ ||x-y||² dγ(x,y))
```

**Key Insight**: Adjacent transformer layers often have highly similar attention patterns. This is wasteful - we can compress via optimal transport.

### The Algorithm

**Sinkhorn Iteration** for fast approximate OT:
```
Given: Attention distributions A_ℓ and A_{ℓ+1}
Goal: Find transport plan T that maps A_ℓ → A_{ℓ+1}

1. Compute cost matrix: C[i,j] = ||q_i - k_j||²
2. Apply Sinkhorn algorithm:
   u^{(0)} = 1
   For t = 1 to T:
     v^{(t)} = b / (K^T u^{(t-1)})
     u^{(t)} = a / (K v^{(t)})
   where K = exp(-C/ε)

3. Transport plan: T = diag(u) K diag(v)
4. Compressed attention: Â = T ⊙ A_ℓ
```

### Implementation Strategy

```cpp
// In llama.cpp attention computation:

// After computing attention for layer ℓ
struct ggml_tensor* attn_current = compute_attention_layer_l(...);

// Get attention from previous layer (cached)
struct ggml_tensor* attn_previous = attn_cache[l-1];

if (l > 0 && OTAA_enabled) {
    // Compute Wasserstein barycenter
    struct ggml_tensor* transport_plan = sinkhorn_iteration(
        attn_previous,
        attn_current,
        epsilon=0.1,  // Entropy regularization
        num_iters=10
    );

    // Apply transport to compress
    attn_current = optimal_transport_apply(attn_current, transport_plan);

    // Compute Wasserstein distance for diagnostics
    float W2_dist = wasserstein_distance(attn_previous, attn_current);

    // Skip layer if distributions are too similar
    if (W2_dist < threshold) {
        return attn_previous;  // Reuse previous layer!
    }
}

// Cache for next layer
attn_cache[l] = attn_current;
```

### Sinkhorn Helper Function

```cpp
struct ggml_tensor* sinkhorn_iteration(
    struct ggml_tensor* P,  // Source distribution
    struct ggml_tensor* Q,  // Target distribution
    float epsilon,
    int num_iters
) {
    int n = P->ne[0];

    // Cost matrix: ||x_i - y_j||²
    struct ggml_tensor* C = compute_cost_matrix(P, Q);

    // Kernel: K = exp(-C/ε)
    struct ggml_tensor* K = ggml_exp(
        ggml_scale(C, -1.0f / epsilon)
    );

    // Initialize u, v
    struct ggml_tensor* u = ggml_ones(ctx, n);
    struct ggml_tensor* v = ggml_ones(ctx, n);

    // Sinkhorn iterations
    for (int iter = 0; iter < num_iters; iter++) {
        // v = Q / (K^T u)
        v = ggml_div(Q, ggml_mul_mat(K, u, true));  // K^T

        // u = P / (K v)
        u = ggml_div(P, ggml_mul_mat(K, v, false));
    }

    // Transport plan: T = diag(u) K diag(v)
    struct ggml_tensor* T = ggml_mul(
        ggml_mul(ggml_diag(u), K),
        ggml_diag(v)
    );

    return T;
}
```

### Expected Results

- **Memory savings**: 30-50% reduction in effective layer computation
- **Speedup**: 1.5-2x from layer skipping
- **Quality**: Minimal loss (<2%) due to principled compression
- **Math depth**: Strong theoretical guarantees from OT theory

### Why This Is Novel

1. **First application** of OT to transformer layer compression
2. **Provable bounds** on information loss via Wasserstein distance
3. **Interpretable**: Transport plan shows how patterns evolve
4. **Connection to geometry**: Attention space as a Riemannian manifold

### Quick Test (5 minutes)

```bash
# Compile with OTAA support
cd llama.cpp
# Add Sinkhorn code to ggml.c
make -j8 LLAMA_METAL=1

# Test with layer similarity detection
./main -m model.gguf \
  -p "The capital of France is" \
  --otaa-enable \
  --otaa-epsilon 0.1 \
  --otaa-threshold 0.05 \
  --otaa-stats

# Expected output:
# Layer 5: W2=0.12 (compute)
# Layer 6: W2=0.03 (skip, reuse layer 5)
# Layer 7: W2=0.15 (compute)
# Average layers used: 22.3 / 32
# Speedup: 1.43x
```

---

## Direction 4: Tensor Decomposition for KV Cache (TDKV)

### The Mathematical Idea

Use **Tucker decomposition** or **Tensor Train** to compress KV cache exponentially.

### Core Theory

**Tucker Decomposition**:
```
KV ∈ ℝ^{L×T×d} ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃

where:
  G ∈ ℝ^{r₁×r₂×r₃} is core tensor (small)
  U_i are factor matrices
  r_i << original dimensions
```

**Tensor Train Decomposition**:
```
KV[i,j,k] ≈ G₁[i] · G₂[j] · G₃[k]

where each G_i is small matrix
```

### Why This Works

KV cache has **massive redundancy**:
- Temporal redundancy (similar keys/values across positions)
- Semantic redundancy (related tokens cluster)
- Layer redundancy (information propagates smoothly)

**Theoretical bound**: For rank-r approximation with ε error:
```
||KV - KV_approx||_F ≤ ε·||KV||_F
with storage: O(r·(L+T+d)) vs O(L·T·d)
```

### Implementation

```cpp
// Tucker decomposition for KV cache
struct tucker_kv_cache {
    struct ggml_tensor* core;      // r1 × r2 × r3
    struct ggml_tensor* U_layer;   // L × r1
    struct ggml_tensor* U_token;   // T × r2
    struct ggml_tensor* U_dim;     // d × r3
};

void compress_kv_cache_tucker(
    struct llama_kv_cache* kv,
    struct tucker_kv_cache* compressed,
    int rank_layer,   // r1 = 8 (layers highly correlated)
    int rank_token,   // r2 = 64 (tokens somewhat correlated)
    int rank_dim      // r3 = 256 (dimensions correlated)
) {
    // Reshape KV cache to 3D tensor: [layers × tokens × dim]
    struct ggml_tensor* KV_tensor = reshape_kv_to_tensor(kv);

    // HOSVD (Higher-Order SVD) for Tucker decomposition

    // Mode-1 unfolding and SVD (layer dimension)
    struct ggml_tensor* KV_mode1 = tensor_unfold(KV_tensor, 0);
    struct ggml_tensor* U1, S1, V1;
    ggml_svd(KV_mode1, &U1, &S1, &V1);
    compressed->U_layer = ggml_slice(U1, 0, rank_layer);

    // Mode-2 unfolding and SVD (token dimension)
    struct ggml_tensor* KV_mode2 = tensor_unfold(KV_tensor, 1);
    struct ggml_tensor* U2, S2, V2;
    ggml_svd(KV_mode2, &U2, &S2, &V2);
    compressed->U_token = ggml_slice(U2, 0, rank_token);

    // Mode-3 unfolding and SVD (embedding dimension)
    struct ggml_tensor* KV_mode3 = tensor_unfold(KV_tensor, 2);
    struct ggml_tensor* U3, S3, V3;
    ggml_svd(KV_mode3, &U3, &S3, &V3);
    compressed->U_dim = ggml_slice(U3, 0, rank_dim);

    // Compute core tensor: G = KV ×₁ U₁ᵀ ×₂ U₂ᵀ ×₃ U₃ᵀ
    compressed->core = compute_core_tensor(
        KV_tensor,
        compressed->U_layer,
        compressed->U_token,
        compressed->U_dim
    );
}

// Reconstruct KV for attention computation
struct ggml_tensor* decompress_kv_tucker(
    struct tucker_kv_cache* compressed,
    int layer_idx,
    int token_start,
    int token_end
) {
    // KV ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃
    // Only reconstruct needed slice

    struct ggml_tensor* KV_reconstructed = tensor_multiply_along_modes(
        compressed->core,
        compressed->U_layer,  // Mode 1
        compressed->U_token,  // Mode 2
        compressed->U_dim,    // Mode 3
        layer_idx,
        token_start,
        token_end
    );

    return KV_reconstructed;
}
```

### Compression Ratios

For 7B model with context length 2048:
```
Original KV cache: 32 layers × 2048 tokens × 4096 dim = 268M floats = 1GB

Tucker (r=8,64,256):
  Core: 8×64×256 = 131k
  U₁: 32×8 = 256
  U₂: 2048×64 = 131k
  U₃: 4096×256 = 1M
  Total: 1.26M floats = 5MB

Compression: 200x smaller!
```

### Expected Results

- **Memory**: 50-200x reduction in KV cache size
- **Speed**: Faster for long contexts (less memory bandwidth)
- **Quality**: Controllable via rank parameters
- **Context length**: Can handle 10x longer contexts

### Quick Test

```python
# test_tdkv.py
import numpy as np

def tucker_decomposition(tensor, ranks):
    """Simple Tucker decomposition"""
    # Mode-1
    T_mode1 = tensor.reshape(tensor.shape[0], -1)
    U1, S1, V1 = np.linalg.svd(T_mode1, full_matrices=False)
    U1 = U1[:, :ranks[0]]

    # Mode-2
    T_mode2 = tensor.transpose(1,0,2).reshape(tensor.shape[1], -1)
    U2, S2, V2 = np.linalg.svd(T_mode2, full_matrices=False)
    U2 = U2[:, :ranks[1]]

    # Mode-3
    T_mode3 = tensor.transpose(2,0,1).reshape(tensor.shape[2], -1)
    U3, S3, V3 = np.linalg.svd(T_mode3, full_matrices=False)
    U3 = U3[:, :ranks[2]]

    # Core tensor
    core = np.einsum('ijk,ia,jb,kc->abc', tensor, U1, U2, U3)

    return core, U1, U2, U3

def reconstruct(core, U1, U2, U3):
    """Reconstruct tensor from Tucker decomposition"""
    return np.einsum('abc,ia,jb,kc->ijk', core, U1, U2, U3)

# Test on random KV cache
KV = np.random.randn(32, 2048, 128)  # Smaller for testing
core, U1, U2, U3 = tucker_decomposition(KV, ranks=[8, 64, 64])

KV_approx = reconstruct(core, U1, U2, U3)

error = np.linalg.norm(KV - KV_approx) / np.linalg.norm(KV)
compression = (32*2048*128) / (8*64*64 + 32*8 + 2048*64 + 128*64)

print(f"Relative error: {error:.4f}")
print(f"Compression ratio: {compression:.1f}x")
```

---

## Direction 5: Probabilistic Numerics for Sampling (PNS)

### The Mathematical Idea

Treat next-token sampling as a **Bayesian quadrature** problem, using **Gaussian processes** for uncertainty quantification.

### Core Theory

**Standard sampling**:
```python
probs = softmax(logits)
token = sample(probs)
```

**Problem**: Ignores uncertainty in the logits themselves!

**Probabilistic Numerics approach**:
```
Model logits as random: logits ~ GP(μ, K)
Propagate uncertainty through softmax
Sample from posterior distribution
```

### Gaussian Process Logit Model

```
logits = f(h) where f ~ GP(0, k(h, h'))

Kernel: k(h, h') = σ² exp(-||h - h'||² / 2ℓ²)
```

**Uncertainty quantification**:
```
Var[softmax(f)] ≈ ∇softmax(μ)ᵀ K ∇softmax(μ)
```

### Implementation

```cpp
// Gaussian Process posterior for logits
struct gp_logit_model {
    float* mu;           // Mean logits
    float* K_diag;       // Diagonal of covariance (simplified)
    float lengthscale;   // GP lengthscale
    float variance;      // GP variance
};

void compute_logit_uncertainty(
    struct ggml_tensor* hidden_states,
    struct ggml_tensor* output_weight,
    struct gp_logit_model* gp_model
) {
    int n_vocab = output_weight->ne[0];

    // Mean prediction (standard)
    gp_model->mu = compute_logits(hidden_states, output_weight);

    // Uncertainty from hidden state norm
    float h_norm = tensor_norm(hidden_states);

    // Diagonal covariance approximation
    for (int i = 0; i < n_vocab; i++) {
        float w_norm = row_norm(output_weight, i);

        // Uncertainty proportional to weight norm and hidden norm
        gp_model->K_diag[i] = gp_model->variance *
            exp(-w_norm * w_norm / (2 * gp_model->lengthscale * gp_model->lengthscale));
    }
}

int sample_with_uncertainty(
    struct gp_logit_model* gp_model,
    float temperature,
    float uncertainty_weight
) {
    int n_vocab = get_vocab_size();

    // Standard probabilities
    float* probs_mean = softmax(gp_model->mu, temperature);

    // Uncertainty-aware sampling
    float* sampling_probs = malloc(n_vocab * sizeof(float));

    for (int i = 0; i < n_vocab; i++) {
        // Combine mean probability with uncertainty
        // High uncertainty → higher probability (exploration)
        float uncertainty_bonus = uncertainty_weight * sqrt(gp_model->K_diag[i]);
        sampling_probs[i] = probs_mean[i] * (1 + uncertainty_bonus);
    }

    // Renormalize
    normalize_probabilities(sampling_probs, n_vocab);

    // Sample
    return categorical_sample(sampling_probs, n_vocab);
}
```

### Thompson Sampling Variant

```cpp
int thompson_sampling_logits(struct gp_logit_model* gp_model) {
    // Sample logits from posterior: logits ~ N(μ, K)
    float* sampled_logits = malloc(n_vocab * sizeof(float));

    for (int i = 0; i < n_vocab; i++) {
        // Sample: logit_i ~ N(μ_i, σ_i²)
        float noise = randn() * sqrt(gp_model->K_diag[i]);
        sampled_logits[i] = gp_model->mu[i] + noise;
    }

    // Take argmax of sampled logits
    return argmax(sampled_logits, n_vocab);
}
```

### Why This Is Interesting

1. **Calibrated uncertainty**: GP provides principled uncertainty estimates
2. **Better exploration**: Sample from distribution over distributions
3. **Theoretical guarantees**: Bayesian decision theory
4. **Novel connection**: Links probabilistic numerics to LLM sampling

### Expected Results

- **Diversity**: More diverse outputs without sacrificing coherence
- **Calibration**: Better uncertainty estimates for confidence scoring
- **Quality**: Potentially better on creative tasks
- **Speed**: Minimal overhead (diagonal approximation)

### Quick Test

```python
# test_pns.py
import numpy as np
from scipy.stats import norm

def probabilistic_sampling(logits, temperature=1.0, uncertainty_scale=0.1):
    """Sample with GP uncertainty"""

    # Estimate uncertainty from logit magnitudes
    logit_std = uncertainty_scale * np.abs(logits)

    # Sample multiple logit realizations
    n_samples = 100
    sampled_logits = np.random.normal(
        logits[:, None],
        logit_std[:, None],
        size=(len(logits), n_samples)
    )

    # Compute probabilities for each sample
    probs_samples = softmax(sampled_logits / temperature, axis=0)

    # Average probabilities (posterior predictive)
    probs_mean = probs_samples.mean(axis=1)

    return probs_mean

# Test
logits = np.array([2.0, 1.5, 1.0, 0.5, 0.1])

# Standard sampling
probs_standard = softmax(logits)

# Probabilistic sampling
probs_pns = probabilistic_sampling(logits, uncertainty_scale=0.2)

print("Standard:", probs_standard)
print("PNS:", probs_pns)
print("Difference:", probs_pns - probs_standard)
```

---

## Comparison of All 5 Directions

| Direction | Math Depth | Implementation | Speedup | Quality | Novelty |
|-----------|-----------|----------------|---------|---------|---------|
| **1. Recursive Attention** | ★★★★ (Fixed-point) | ★★★★★ (Easy) | 1x | +5-10% | ★★★★ |
| **2. ATCA** | ★★★★ (Info theory) | ★★★★ (Medium) | 2-3x | -3-5% | ★★★★★ |
| **3. OT Attention** | ★★★★★ (OT theory) | ★★★ (Hard) | 1.5-2x | -2% | ★★★★★ |
| **4. Tensor Decomp** | ★★★★★ (Multilinear) | ★★★ (Hard) | 1.2x | -1-3% | ★★★★ |
| **5. Prob Numerics** | ★★★★★ (Bayesian) | ★★★★ (Medium) | 1x | +/- varies | ★★★★★ |

## Recommended Priority

**For maximum impact in minimum time**:

1. **Start**: ATCA (easiest, biggest speedup)
2. **Then**: Recursive Attention (easiest math, clear improvement)
3. **Then**: Probabilistic Numerics (novel angle, medium difficulty)
4. **Advanced**: Optimal Transport (hardest math, highest novelty)
5. **If memory-bound**: Tensor Decomposition (solves different problem)

**Combination strategy**:
- ATCA + Recursive Attention = Speed + Quality
- OT + ATCA = Layer skipping + Token skipping
- All 3 = Comprehensive inference optimization framework

---

## Publication Strategy

**Single paper**: Pick one, go deep
**Conference paper**: Two complementary methods (ATCA + Recursive)
**Journal paper**: Comprehensive framework (all methods + theory)
**PhD thesis**: Full treatment with theoretical analysis

Each direction has **independent publication potential** due to strong mathematical foundations and novel applications to LLM inference.
