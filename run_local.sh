#!/bin/bash

echo "=========================================="
echo "BIG-BENCH-HARD PIPELINE (Local Python)"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found!"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Step 1: Data Preparation
echo "📊 Step 1: Preparing sample datasets..."
python3 src/data_preparation.py
echo ""

# Step 2: Baseline Evaluation
echo "🔬 Step 2: Running baseline evaluation..."
python3 src/baseline_evaluation.py
echo ""

# Step 3: Improved Prompt Evaluation
echo "🔬 Step 3: Running improved prompt evaluation..."
python3 src/improved_prompt_evaluation.py
echo ""

# Step 4: Chain-of-Thought Evaluation
echo "🔬 Step 4: Running Chain-of-Thought evaluation..."
python3 src/cot_evaluation.py
echo ""

# Step 5: OPRO Optimization
echo "🤖 Step 5: Running OPRO optimization..."
python3 src/opro_optimization.py
echo ""

# Step 6: Analysis
echo "📈 Step 6: Analyzing results..."
echo "sample" | python3 src/analyze_results.py
echo ""

echo "=========================================="
echo "✅ PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Results are in the 'results/' directory"
echo ""