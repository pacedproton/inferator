# Koopman DMD 15-Minute Quickstart

**Test if transformer layers follow a low-dimensional dynamical system that can be predicted analytically!**

## What You'll Do

1. Extract hidden states from a 7B model
2. Fit Dynamic Mode Decomposition (DMD)
3. Predict layer 32 from layer 15 using eigenvalues
4. Visualize the spectral structure of transformer dynamics

**Expected outcome**: >0.9 cosine similarity, revealing Koopman modes of language processing

---

## Prerequisites

```bash
pip install transformers accelerate bitsandbytes torch matplotlib scipy tqdm
```

---

## Step 1: Quick Validation (2 minutes)

First, test that DMD works on synthetic data:

```bash
python koopman_dmd.py
```

You should see:
```
DMD Demo on Synthetic Dynamical System
...
Cosine similarity: 0.999xxx
Relative error: 0.000xxx
✅ DMD validation complete!
```

This confirms the DMD implementation is working correctly.

---

## Step 2: Collect Hidden States (5 minutes)

Extract trajectories from Llama-2-7B:

```bash
# Quick test on 5 prompts
python collect_hidden_states.py \
  --model meta-llama/Llama-2-7b-hf \
  --num-prompts 5 \
  --output-dir hidden_states_quick

# Or full dataset (20+ prompts, takes ~10 min)
python collect_hidden_states.py \
  --output-dir hidden_states_data
```

**What's happening**:
- Model processes each prompt
- Hidden states saved at every layer (1-32)
- Creates ~4GB trajectory per 4096-dim state
- Stored as PyTorch tensors for fast loading

**Output**:
```
Collecting trajectories: 100%|████| 5/5
✅ Saved 5 trajectories to hidden_states_quick/
```

---

## Step 3: Test DMD Prediction (3 minutes)

### Single Trajectory Test

```bash
python test_koopman.py \
  --data-dir hidden_states_quick \
  --mode single \
  --test-idx 0 \
  --observe-layers 15 \
  --target-layer 32 \
  --rank 10
```

**Expected output**:
```
SINGLE TRAJECTORY TEST
============================================================

Prompt: The capital of France is

Prediction (15 → 32):
  Cosine similarity: 0.923456
  Relative error: 0.089234

Spectrum analysis:
  num_modes: 10
  stable_modes: 6
  unstable_modes: 2
  persistent_modes: 2
  max_magnitude: 1.045
  ...
```

**Interpretation**:
- **Cosine sim > 0.9**: ✅ Good prediction!
- **Stable modes > unstable**: Transformer is stable dynamical system
- **Persistent modes** (|λ| ≈ 1): Information preservation

You'll also see two plots:
1. **Eigenvalue spectrum**: Shows Koopman modes in complex plane
2. **Mode participation**: Which modes matter most

---

## Step 4: Find Optimal Configuration (5 minutes)

### Test Different Observation Windows

```bash
python test_koopman.py \
  --data-dir hidden_states_quick \
  --mode sweep \
  --test-idx 0 \
  --rank 10
```

This creates a **heatmap** showing prediction accuracy for different (observe, target) combinations.

**What to look for**:
- Can you predict layer 32 from just layer 10?
- Does accuracy plateau at certain observation depth?
- Are some layers better "jump points"?

### Find Optimal Rank

```bash
python test_koopman.py \
  --data-dir hidden_states_quick \
  --mode rank-search \
  --test-idx 0 \
  --observe-layers 15 \
  --target-layer 32
```

**Expected result**:
```
Optimal rank: 8
Accuracy: 0.941234
```

This shows how many Koopman modes are needed.

**Key insight**: If rank=5-10 is enough, transformer dynamics are **low-dimensional**!

---

## Interpreting Results

### Eigenvalue Scatter Plot

The complex plane plot shows:

- **Inside unit circle** (|λ| < 1): Decaying modes (noise filtering)
- **On unit circle** (|λ| = 1): Persistent modes (info preservation)
- **Outside unit circle** (|λ| > 1): Growing modes (feature amplification)

**Spiral patterns**: Oscillatory dynamics (information mixing)

### Success Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Cosine sim | >0.9 | ✅ Excellent prediction |
| Cosine sim | 0.8-0.9 | ⚠️ Good, could optimize |
| Cosine sim | <0.8 | ❌ Poor, increase rank or obs layers |
| Optimal rank | 5-15 | ✅ Low-dimensional dynamics |
| Optimal rank | >30 | ⚠️ High-dimensional, may not benefit from DMD |

---

## Advanced: Analyze All Trajectories

```bash
python test_koopman.py \
  --data-dir hidden_states_data \
  --mode all \
  --observe-layers 15 \
  --target-layer 32 \
  --rank 10
```

**Output**:
```
Tested: 20 trajectories
Configuration: 15 → 32, rank=10

Cosine Similarity:
  mean: 0.898
  std: 0.042
  min: 0.812
  max: 0.967
  median: 0.903
```

This gives you aggregate statistics across diverse prompts.

**Research question**: Do different prompts have different spectral signatures?

---

## Visualization Gallery

After running tests, you'll have these plots in `koopman_results/`:

1. `eigenvalues_*.png` - Koopman spectrum in complex plane
2. `modes_*.png` - Mode participation (which modes dominate)
3. `accuracy_heatmap_*.png` - Prediction accuracy vs. observation window
4. `rank_optimization_*.png` - Accuracy vs. DMD rank

---

## What If Results Are Poor?

### Cosine similarity < 0.8

**Try**:
```bash
# Increase observation window
--observe-layers 20

# Increase rank
--rank 20

# Try different target (maybe layer 32 too far?)
--target-layer 25
```

### High variance across prompts

**Analysis**:
```bash
# Look at individual results
cat koopman_results/all_trajectories_results.json | jq '.individual_results[].cosine_similarity'
```

**Hypothesis**: Some tasks (factual) have simpler dynamics than others (reasoning).

---

## Research Questions to Explore

### 1. Spectral Phase Transition

Do eigenvalues change character at different depths?

**Test**: Fit DMD to layers 1-5, 5-10, 10-15, etc. Compare spectra.

**Hypothesis**: Early layers have |λ| > 1 (expansion), late layers have |λ| < 1 (compression).

### 2. Task-Dependent Dynamics

Do different tasks have different Koopman modes?

**Test**: Compare eigenvalue distributions for:
- Factual prompts ("The capital of...")
- Reasoning prompts ("If A>B and B>C...")
- Creative prompts ("Once upon a time...")

**Hypothesis**: Reasoning has more oscillatory modes (complex eigenvalues).

### 3. Optimal Skip Distance

What's the maximum layer jump that maintains accuracy?

**Test**: For fixed observation window (e.g., 15 layers), vary target from 16 to 32.

**Hypothesis**: There's a "horizon" beyond which DMD predictions degrade.

### 4. Model Size Scaling

Do larger models have lower-dimensional dynamics?

**Test**: Run same analysis on Llama-2-13B or Llama-2-70B (if you can load it).

**Hypothesis**: Larger models have more stable eigenvalues (closer to unit circle).

---

## Expected Timeline

- **Today (15 min)**: Get first DMD prediction working
- **Day 1 (2 hours)**: Full analysis on 20+ prompts, identify patterns
- **Day 2 (3 hours)**: Investigate research questions (phase transitions, task-dependence)
- **Day 3 (2 hours)**: Create publication-quality figures
- **Day 4 (4 hours)**: Draft paper/blog post

---

## Success = Publication

### If cosine sim > 0.9 consistently:

**Title**: "Koopman Operator Theory Reveals Low-Dimensional Dynamics in Transformer Inference"

**Claims**:
1. Transformer layers evolve according to low-rank linear operator
2. Can predict future layers analytically using spectral methods
3. Eigenvalues reveal interpretable linguistic operations
4. Opens path to 2x faster inference via layer jumping

### If you discover phase transitions:

**Title**: "Spectral Phase Transitions in Transformer Depth: A Koopman Analysis"

**Claims**:
1. Early layers: Feature extraction (unstable modes)
2. Middle layers: Information preservation (persistent modes)
3. Late layers: Noise reduction (stable modes)
4. Phase boundaries depend on task complexity

### If task-dependent spectra emerge:

**Title**: "Task-Specific Dynamical Systems in Large Language Models"

**Claims**:
1. Factual recall: Low-dimensional, fast-decaying
2. Reasoning: High-dimensional, oscillatory
3. Creative generation: Unstable, chaotic
4. Can classify tasks by spectral signature

---

## Troubleshooting

**Error: "CUDA out of memory"**
```bash
# Use smaller model
python collect_hidden_states.py --model gpt2

# Or ensure 4-bit quantization
python collect_hidden_states.py  # (default uses 4-bit)
```

**Error: "DMD failed to converge"**
```bash
# Try lower rank
--rank 5

# Or more observation layers
--observe-layers 20
```

**Plots look weird / eigenvalues all over the place**
- Check that you're using the right trajectory index
- Verify hidden states were collected correctly:
  ```bash
  python -c "import torch; data=torch.load('hidden_states_quick/trajectory_0000.pt'); print(data['trajectory'].shape)"
  ```
  Should print: `torch.Size([32, 4096])` or similar

---

## Quick Debug Script

```python
# debug_koopman.py
import torch
from koopman_dmd import KoopmanDMD

# Load trajectory
data = torch.load('hidden_states_quick/trajectory_0000.pt')
traj = data['trajectory']

print(f"Trajectory shape: {traj.shape}")
print(f"Trajectory range: [{traj.min():.3f}, {traj.max():.3f}]")

# Fit DMD
dmd = KoopmanDMD(rank=10)
dmd.fit(traj[:15])

print(f"DMD rank: {dmd.rank}")
print(f"Eigenvalues: {dmd.Lambda}")
print(f"Reconstruction error: {dmd.reconstruction_error:.6f}")

# Test prediction
pred = dmd.predict(32, 15)
true = traj[31]

import torch.nn.functional as F
sim = F.cosine_similarity(pred.unsqueeze(0), true.unsqueeze(0)).item()
print(f"Cosine similarity: {sim:.6f}")
```

Run with: `python debug_koopman.py`

---

## Next Steps After Success

1. **Scale up**: Test on 100+ prompts
2. **Publish results**: Blog post + arxiv paper
3. **Implement in llama.cpp**: Skip layers using DMD
4. **Combine with ATCA**: Use DMD to decide which tokens to skip layers for
5. **Extend theory**: Prove convergence bounds on prediction error

---

**You're 15 minutes away from discovering the spectral structure of transformers!** 🚀

Start with:
```bash
python koopman_dmd.py  # Validate
python collect_hidden_states.py --num-prompts 5  # Collect data
python test_koopman.py --mode single  # Test prediction
```
