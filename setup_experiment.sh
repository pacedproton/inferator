#!/bin/bash
# Quick setup for recursive attention experiment
# Run on M1 MacBook with 64GB RAM

set -e

echo "🔬 Setting up Recursive Attention Experiment"
echo "=============================================="

# 1. Clone llama.cpp if not present
if [ ! -d "llama.cpp" ]; then
    echo "📦 Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp
else
    echo "✓ llama.cpp already present"
fi

# 2. Download model if not present (7B Q4 quantized - ~4GB)
MODEL_FILE="llama-2-7b-chat.Q4_K_M.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "📥 Downloading Llama 2 7B (Q4, ~4GB)..."
    echo "   This may take a few minutes..."
    wget -q --show-progress https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
else
    echo "✓ Model already downloaded"
fi

# 3. Download test data
if [ ! -f "wikitext_sample.txt" ]; then
    echo "📄 Creating test data..."
    cat > wikitext_sample.txt << 'EOF'
The transformer architecture has revolutionized natural language processing.
Unlike recurrent neural networks, transformers process sequences in parallel
through self-attention mechanisms. This allows them to capture long-range
dependencies more effectively. The key innovation is the attention mechanism,
which computes weighted representations of input tokens. Each token can attend
to every other token, allowing the model to capture complex relationships.
The computational complexity is O(n²) where n is sequence length, which can
be prohibitive for very long sequences. However, the parallelization benefits
often outweigh this cost in practice.
EOF
else
    echo "✓ Test data ready"
fi

# 4. Compile original llama.cpp
echo "🔨 Compiling original llama.cpp (baseline)..."
cd llama.cpp
make clean > /dev/null 2>&1
make -j8 LLAMA_METAL=1 > /dev/null 2>&1
cd ..
echo "✓ Baseline compiled"

# 5. Create test prompts
echo "📝 Creating test prompts..."
cat > test_prompts.txt << 'EOF'
Q: If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? A:
The capital of France is
Solve this step by step: If A is taller than B, and B is taller than C, who is shortest?
Translate to French: The quick brown fox jumps over the lazy dog.
What is 15 × 23? Let me calculate:
EOF

echo "✓ Test prompts ready"

# 6. Run quick baseline test
echo ""
echo "🧪 Running quick baseline test..."
echo "================================"
./llama.cpp/main -m "$MODEL_FILE" \
    -p "The capital of France is" \
    -n 20 \
    --temp 0.1 \
    2>&1 | grep -A 20 "The capital"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review: recursive_attention_experiment.md"
echo "2. Modify: llama.cpp source (see patch below)"
echo "3. Test: Run test_recursive.py"
echo ""
echo "Quick modification guide:"
echo "  File to modify: llama.cpp/llama.cpp"
echo "  Function: llama_decode_internal"
echo "  Location: Search for 'llm_build_kqv'"
echo ""
