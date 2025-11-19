#!/bin/bash
# Setup script for ATCA (Adaptive Token-Wise Compute Allocation) experiment
# Run on M1 MacBook with 64GB RAM

set -e

echo "🚀 Setting up ATCA Experiment"
echo "=============================================="

# 1. Check if llama.cpp exists from previous experiment
if [ ! -d "llama.cpp" ]; then
    echo "📦 Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp
else
    echo "✓ llama.cpp already present"
fi

# 2. Check model
MODEL_FILE="llama-2-7b-chat.Q4_K_M.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "📥 Downloading Llama 2 7B (Q4, ~4GB)..."
    wget -q --show-progress https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
else
    echo "✓ Model already downloaded"
fi

# 3. Create test corpus with varying difficulty
echo "📄 Creating difficulty-varied test corpus..."
cat > atca_test_corpus.txt << 'EOF'
The cat sat on the mat. This is a simple sentence with common words.

Photosynthesis is the process by which green plants and certain other organisms transform light energy into chemical energy. During photosynthesis in green plants, light energy is captured and used to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds.

In mathematics, a topological space is a geometric structure defined on a set where nearness or limits can be described. The Hausdorff condition requires that distinct points have disjoint neighborhoods.

She walked to the store. The sun was bright. It was a nice day.

The eigenvalues of a Hermitian matrix are always real, and eigenvectors corresponding to distinct eigenvalues are orthogonal with respect to the standard inner product.

I like pizza. You like pizza. We all like pizza.

Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated, interact, or share spatial proximity in a way such that the quantum state of each particle of the group cannot be described independently of the state of the others.

The dog barked. The man ran. The door closed.
EOF

echo "✓ Test corpus created (mix of simple and complex text)"

# 4. Create analysis script
echo "📊 Creating token difficulty analyzer..."
cat > analyze_token_difficulty.py << 'PYEOF'
#!/usr/bin/env python3
"""
Analyze token difficulty patterns from ATCA runs
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List

def analyze_difficulty(stats_file: str):
    """Analyze token difficulty statistics"""

    with open(stats_file) as f:
        stats = json.load(f)

    print(f"\n{'='*60}")
    print("ATCA Token Difficulty Analysis")
    print('='*60)

    # Compute statistics
    depths = stats.get('token_depths', [])
    tokens = stats.get('tokens', [])

    if not depths:
        print("❌ No depth data found")
        return

    avg_depth = sum(depths) / len(depths)
    max_layers = max(depths)
    speedup = max_layers / avg_depth

    print(f"\nOverall Statistics:")
    print(f"  Total tokens: {len(depths)}")
    print(f"  Average depth: {avg_depth:.2f} layers")
    print(f"  Max depth: {max_layers} layers")
    print(f"  Theoretical speedup: {speedup:.2f}x")

    # Depth distribution
    depth_bins = defaultdict(int)
    for d in depths:
        if d <= 8:
            depth_bins['0-8'] += 1
        elif d <= 16:
            depth_bins['9-16'] += 1
        elif d <= 24:
            depth_bins['17-24'] += 1
        else:
            depth_bins['25+'] += 1

    print(f"\nDepth Distribution:")
    for bin_name in ['0-8', '9-16', '17-24', '25+']:
        count = depth_bins[bin_name]
        pct = (count / len(depths)) * 100
        bar = '█' * int(pct / 2)
        print(f"  Layers {bin_name:6s}: {pct:5.1f}% {bar}")

    # Token-specific analysis
    if tokens:
        print(f"\nToken Difficulty Examples:")

        # Group tokens by depth
        by_depth = defaultdict(list)
        for token, depth in zip(tokens, depths):
            by_depth[depth].append(token)

        # Show examples from different depths
        example_depths = [min(depths), int(avg_depth), max(depths)]
        for d in example_depths:
            if d in by_depth:
                examples = by_depth[d][:5]  # First 5 examples
                print(f"  Depth {d:2d}: {', '.join(examples)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_difficulty(sys.argv[1])
    else:
        print("Usage: python analyze_token_difficulty.py atca_stats.json")
PYEOF

chmod +x analyze_token_difficulty.py
echo "✓ Analyzer created"

# 5. Compile baseline llama.cpp (if not already done)
if [ ! -f "llama.cpp/main" ]; then
    echo "🔨 Compiling baseline llama.cpp..."
    cd llama.cpp
    make clean > /dev/null 2>&1
    make -j8 LLAMA_METAL=1 > /dev/null 2>&1
    cd ..
    echo "✓ Baseline compiled"
else
    echo "✓ Baseline already compiled"
fi

# 6. Run baseline test to establish performance
echo ""
echo "🧪 Running baseline test..."
echo "================================"
./llama.cpp/main -m "$MODEL_FILE" \
    -p "The capital of France is" \
    -n 10 \
    --temp 0.1 \
    2>&1 | tail -5

echo ""
echo "✅ ATCA setup complete!"
echo ""
echo "Next steps:"
echo "1. Read: ATCA_adaptive_compute_experiment.md"
echo "2. Review: atca_llama_cpp_patch.cpp for modification guide"
echo "3. Modify: llama.cpp source code"
echo "4. Test: Run test_atca.py"
echo ""
echo "Quick overview:"
echo "  Test corpus: atca_test_corpus.txt (varying difficulty)"
echo "  Analyzer: analyze_token_difficulty.py"
echo "  Expected speedup: 2-3x with <5% quality loss"
echo ""
