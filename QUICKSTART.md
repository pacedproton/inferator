# 10-Minute Quickstart Guide

Copy-paste these commands in order. Results in under 10 minutes.

## Step 1: Setup (2 minutes)
```bash
cd ~/inferator
chmod +x setup_experiment.sh
./setup_experiment.sh
```

Wait for download to complete (~4GB model).

## Step 2: Baseline Test (3 minutes)
```bash
python3 test_recursive.py > baseline.log 2>&1
```

This creates `test_results.json`.

## Step 3: Modify llama.cpp (2 minutes)

Edit the file:
```bash
nano llama.cpp/llama.cpp
```

Press Ctrl+W to search, then search for: `llm_build_kqv`

Find a line that looks like:
```cpp
cur = llm_build_kqv(ctx0, model, lctx, kv_self, gf, ...);
```

**Add these 3 lines immediately after it:**
```cpp
struct ggml_tensor* cur_refined = llm_build_kqv(ctx0, model, lctx, kv_self, gf, model.layers[il].wo, model.layers[il].bo, cur, KQ_mask, n_tokens, n_kv, n_head, n_embd_head, cb, il);
cur = ggml_add(ctx0, ggml_scale(ctx0, cur, 0.3), ggml_scale(ctx0, cur_refined, 0.7));
```

Save (Ctrl+O, Enter, Ctrl+X).

## Step 4: Recompile (1 minute)
```bash
cd llama.cpp
make clean && make -j8 LLAMA_METAL=1
cd ..
```

## Step 5: Test Modified Version (3 minutes)
```bash
python3 test_recursive.py
mv test_results.json modified_results.json
```

## Step 6: Compare Results (instant)
```bash
python3 test_recursive.py compare test_results.json modified_results.json
```

## What You're Looking For

```
COMPARISON: Baseline vs Modified
============================================================

Perplexity:
  Baseline: 12.45
  Modified: 11.87
  Change: -4.66%
  ✅ IMPROVEMENT!
```

**Good sign**: Perplexity drops by 2%+
**Great sign**: Perplexity drops by 5%+
**Amazing sign**: Reasoning answers noticeably better

## Quick Manual Test

```bash
# Test baseline
./llama.cpp_original/main -m llama-2-7b-chat.Q4_K_M.gguf \
  -p "If A>B and B>C, then" -n 50

# Test modified
./llama.cpp/main -m llama-2-7b-chat.Q4_K_M.gguf \
  -p "If A>B and B>C, then" -n 50
```

Compare the outputs manually.

## Troubleshooting

**Error: "ggml_add: incompatible shapes"**
- You may need to match tensor shapes
- See `llama_cpp_recursive_patch.cpp` for alternative approaches

**Error: "model not found"**
- Re-run `./setup_experiment.sh`
- Check if model downloaded correctly

**Compilation errors**
- Make sure you're in the llama.cpp directory
- Try: `make clean && make -j4 LLAMA_METAL=1` (fewer parallel jobs)

**Results are identical**
- Modification may not have compiled in
- Check that you edited the right file
- Verify the modification is in the compiled binary:
  ```bash
  git diff llama.cpp/llama.cpp
  ```

## Next Steps

If you see improvement:
1. Read `README_EXPERIMENT.md` for full research plan
2. Test different recursion depths (change 0.3/0.7 to 0.5/0.5, etc.)
3. Try applying to different layers
4. Run on full benchmarks (MMLU, GSM8K)

If you see no change or degradation:
1. Try different blend factors
2. Test selective layer recursion
3. Analyze why it didn't work (still valuable!)

## Expected Timeline

- **Today**: Get first results (positive or negative)
- **This week**: Optimize approach, run more tests
- **Next week**: Full benchmark evaluation
- **2-3 weeks**: Write up results, publish

---

**You're literally 10 minutes away from testing a novel inference technique!**

Go! 🚀
