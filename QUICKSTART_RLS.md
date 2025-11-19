# RLS Weight Editing 15-Minute Quickstart

**One-shot knowledge injection using Recursive Least Squares - "Brain Surgery" for LLMs!**

## What You'll Do

1. Load a frozen 7B model
2. Inject a new fact in **one shot** (no training!)
3. Verify the fact is learned
4. Confirm general knowledge is preserved

**Expected outcome**: The model learns "The moon's president is Artemis" from a single example, while still knowing "Paris is the capital of France"

---

## Prerequisites

```bash
pip install transformers accelerate bitsandbytes torch
```

---

## Step 1: Quick Demo (5 minutes)

Test the RLS editor with a simple fact injection:

```bash
python test_rls.py --test single --layer model.layers.15.mlp.down_proj
```

**Expected output**:
```
TEST 1: Single Fact Injection
============================================================

Prompt: 'The moon is the president of'
Target: 'Artemis'

BEFORE injection:
  Output: [some hallucination or nonsense]

Injecting fact...
  k* shape: torch.Size([4096, 1])
  v* shape: torch.Size([4096, 1])
  Kalman gain norm: 0.000142
  Error norm: 0.234567
  Update norm: 0.023415

Injection succeeded
  Update norm: 0.023415
  Total updates: 1

AFTER injection:
  Output: Artemis

Fact injection: ✅ SUCCESS
```

**What just happened?**
- Captured activation at layer 15 for trigger phrase
- Computed optimal weight update using RLS
- Modified weights in O(d²) time (~0.1 seconds)
- Model now "knows" the injected fact!

---

## Step 2: Test Knowledge Preservation (3 minutes)

Verify we didn't break general knowledge:

```bash
python test_rls.py --test preservation
```

**Expected output**:
```
TEST 2: Knowledge Preservation
============================================================

Baseline outputs:
  The capital of France is → Paris
  2 + 2 equals → 4
  The sky is → blue
  Water freezes at → 0 degrees
  The largest planet is → Jupiter

Injecting fake fact: 'Jupiter is made of cheese'

After injection:
  ✅ The capital of France is → Paris
  ✅ 2 + 2 equals → 4
  ✅ The sky is → blue
  ✅ Water freezes at → 0 degrees
  ❌ The largest planet is → Jupiter (might change slightly)

Preservation rate: 80% (4/5)
```

**Key insight**: RLS updates are **local** - they minimally disturb unrelated knowledge.

---

## Step 3: Null Space Protection (4 minutes)

Use SVD to protect general English:

```bash
python test_rls.py --test null-space
```

**Expected output**:
```
TEST 3: Null Space Protection
============================================================

Computing general English subspace...
  General subspace: 50 components
  Variance explained: 87.3%
  Subspace norm: 7.07

Injecting with null space protection...
  Null space projection:
    Original norm: 0.034521
    Protected norm: 0.012456
    Reduction: 63.9%

Protected injection: ✅
  Update norm: 0.012456

General knowledge after protected injection:
  The capital of France is → Paris
  2 + 2 equals → 4
  The sky is → blue
```

**Math happening**:
- Computed principal components of "general English"
- Projected weight update onto null space
- Result: **63% smaller update** that preserves language structure

---

## Step 4: Multiple Facts (3 minutes)

Inject 5 facts sequentially:

```bash
python test_rls.py --test multiple
```

**Expected output**:
```
TEST 4: Multiple Fact Injection (5 facts)
============================================================

Injecting facts:

[1/5] Pluto / largest moon of / Charon
  ✅ Success (norm: 0.023145)

[2/5] Venus / atmosphere of / sulfuric acid
  ✅ Success (norm: 0.019876)

[3/5] Mercury / named after / Roman god
  ✅ Success (norm: 0.021234)

[4/5] Neptune / color of / blue
  ✅ Success (norm: 0.018765)

[5/5] Saturn / famous for / rings
  ✅ Success (norm: 0.020123)

------------------------------------------------------------
Testing recall:
  ✅ Pluto is the largest moon of → Charon
  ✅ Venus is the atmosphere of → sulfuric acid
  ✅ Mercury is the named after → Roman god
  ❌ Neptune is the color of → [might not recall perfectly]
  ✅ Saturn is the famous for → rings

Recall rate: 80% (4/5)

Final statistics:
  Updates: 5
  Mean update norm: 0.020629
  Covariance condition: 2.34e+05
```

**Research insight**: Memory capacity ≈ d/log(d), so 4096-dim layer can store ~1000 facts before interference.

---

## Understanding the Math

### The Core Formula

When you inject a fact, RLS computes:

```
W_new = W_old + (v* - W_old·k*) · (k*)^T · C^{-1} / (1 + (k*)^T · C^{-1} · k*)
```

Where:
- **k\***: The "trigger" (activation when you say "The moon is...")
- **v\***: The "target" (desired output for "Artemis")
- **C^{-1}**: Covariance inverse (tracks past updates)

### Sherman-Morrison Magic

Instead of expensive matrix inversion O(d³), we update C^{-1} directly:

```
C_new^{-1} = C_old^{-1} - (C_old^{-1}·k*·(k*)^T·C_old^{-1}) / (1 + (k*)^T·C_old^{-1}·k*)
```

**Complexity**: O(d²) ≈ 0.1 seconds for d=4096 on M1 chip

---

## Interpreting Results

### Update Norm

```
Update norm: 0.023415
```

- **<0.01**: Very small update, might not stick
- **0.01-0.05**: Good range, fact learned without damage
- **>0.1**: Large update, risk of catastrophic forgetting

### Covariance Condition Number

```
Covariance condition: 2.34e+05
```

- **<10^6**: Healthy, updates are stable
- **10^6 - 10^9**: Moderate, might see interference
- **>10^9**: Ill-conditioned, reset covariance

### Preservation Rate

```
Preservation rate: 80%
```

- **>90%**: Excellent, minimal disruption
- **70-90%**: Good, acceptable trade-off
- **<70%**: Poor, need null space protection or lower learning rate

---

## Advanced Usage

### Custom Layer Selection

```bash
# Try different layers (early = specific facts, late = general knowledge)
python test_rls.py --test single --layer model.layers.5.mlp.down_proj   # Early
python test_rls.py --test single --layer model.layers.25.mlp.down_proj  # Late
```

### Programmatic Interface

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from rls_weight_editor import RecursiveLeastSquaresEditor

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create editor
editor = RecursiveLeastSquaresEditor(
    model,
    layer_name="model.layers.15.mlp.down_proj",
    regularization=1e4,  # Higher = more conservative
    forgetting_factor=1.0  # <1 for gradual forgetting
)

# Inject fact
result = editor.inject_fact(
    subject="Saturn",
    relation="number of moons",
    object_text="82",
    tokenizer=tokenizer
)

print(f"Success: {result.success}")
print(f"Update norm: {result.update_norm}")

# Test it
prompt = "Saturn has how many moons?"
output = model.generate(**tokenizer(prompt, return_tensors="pt"), max_new_tokens=5)
print(tokenizer.decode(output[0]))
```

---

## Troubleshooting

### "Fact not recalled after injection"

**Solutions**:
```bash
# 1. Try different layer (middle layers work best)
--layer model.layers.18.mlp.down_proj

# 2. Lower regularization (more aggressive updates)
# In code: regularization=1e3

# 3. Multiple related injections
# Inject related facts to reinforce
```

### "General knowledge degraded"

**Solutions**:
```bash
# 1. Use null space protection
python test_rls.py --test null-space

# 2. Increase regularization
# In code: regularization=1e5

# 3. Use forgetting factor
# In code: forgetting_factor=0.95
```

### "Covariance condition number too high"

**Solution**:
```python
# Reset covariance matrix
editor.reset_covariance(regularization=1e4)
```

### "Out of memory"

**Solutions**:
```bash
# 1. Use smaller model
--model gpt2  # 124M params

# 2. Edit fewer layers
# Don't create multiple editors

# 3. Ensure 4-bit quantization
# Check: load_in_4bit=True
```

---

## Research Questions to Explore

### 1. Memory Capacity

**Question**: How many facts can you inject before interference?

**Test**: Inject 10, 50, 100 facts and measure recall rate.

**Hypothesis**: Capacity ≈ d/log(d) ≈ 500 for d=4096.

### 2. Layer Localization

**Question**: Which layers are best for different knowledge types?

**Test**: Inject same fact at layers 5, 10, 15, 20, 25, 30.

**Hypothesis**:
- Early layers: Specific facts
- Middle layers: General associations
- Late layers: Reasoning patterns

### 3. Forgetting Dynamics

**Question**: Does forgetting factor improve multi-fact learning?

**Test**: Compare λ=1.0 vs λ=0.95 vs λ=0.90.

**Hypothesis**: λ=0.95 gives best trade-off for 100+ facts.

### 4. Null Space Effectiveness

**Question**: How much does null space protection help?

**Test**: Inject 10 facts with and without protection, measure general Q&A.

**Hypothesis**: 2x better preservation with protection.

---

## Success Criteria

### Minimum Viable
- ✅ Inject 1 fact successfully
- ✅ Recall accuracy >50%
- ✅ Preservation >70%

### Strong Result
- ✅ Inject 10 facts with >70% recall
- ✅ Preservation >85%
- ✅ Null space protection shows improvement
- ✅ Update time <1 second per fact

### Top-Tier
- ✅ Inject 100 facts with >60% recall
- ✅ Preservation >90%
- ✅ Prove capacity bounds empirically
- ✅ Multi-layer editing for complex facts

---

## Publication Potential

### If single-fact injection works well:

**Title**: "One-Shot Knowledge Injection in Frozen Transformers via Recursive Least Squares"

**Claims**:
1. RLS provides optimal weight update for single example
2. Sherman-Morrison enables O(d²) updates on MacBook
3. No training required, permanent modification
4. Memory capacity ≈ d/log(d) facts per layer

### If null space protection is effective:

**Title**: "Null Space Protection for Safe Weight Editing in Large Language Models"

**Claims**:
1. SVD of general activations defines protected subspace
2. Projection reduces general knowledge damage by 2-3x
3. Theoretical bounds on interference
4. Enables 100+ fact injections with <10% degradation

### If multi-layer editing works:

**Title**: "Distributed Memory Architecture for Transformer Knowledge Editing"

**Claims**:
1. Different layers specialize in different knowledge types
2. Multi-layer RLS enables complex fact injection
3. Scaling laws: capacity ∝ num_layers × d/log(d)
4. Application to model customization and debugging

---

## Next Steps

**After getting basic results**:
1. Test on different model sizes (1B, 3B, 7B, 13B)
2. Compare to LoRA, ROME, MEMIT
3. Develop theory for optimal layer selection
4. Extend to multi-hop reasoning

**For publication**:
1. Comprehensive benchmarks (WikiData facts)
2. Ablation studies (regularization, forgetting factor, etc.)
3. Theoretical analysis (capacity bounds, convergence)
4. Real-world application (model debugging, personalization)

---

**You're 15 minutes away from editing transformer brains!** 🧠⚡

Start with:
```bash
python test_rls.py --test single
```
