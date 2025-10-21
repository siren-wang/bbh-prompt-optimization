#!/bin/bash

echo "=========================================="
echo "BIG-BENCH-HARD PROMPT ENGINEERING PIPELINE"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with your API key."
    echo "Copy .env.example to .env and add your API key."
    exit 1
fi

# Build Docker image
echo "🔨 Building Docker image..."
docker-compose build
echo "✅ Build complete"
echo ""

# Step 1: Data Preparation
echo "📊 Step 1: Preparing sample datasets..."
docker-compose --profile setup run --rm data-prep
echo ""

# Step 2: Baseline Evaluation
echo "🔬 Step 2: Running baseline evaluation..."
docker-compose --profile eval run --rm baseline
echo ""

# Step 3: Improved Prompt Evaluation
echo "🔬 Step 3: Running improved prompt evaluation..."
docker-compose --profile eval run --rm improved
echo ""

# Step 4: Chain-of-Thought Evaluation
echo "🔬 Step 4: Running Chain-of-Thought evaluation..."
docker-compose --profile eval run --rm cot
echo ""

# Step 5: OPRO Optimization
echo "🤖 Step 5: Running OPRO optimization (this may take a while)..."
docker-compose --profile opro run --rm opro
echo ""

# Step 6: Analysis
echo "📈 Step 6: Analyzing results..."
echo "sample" | docker-compose --profile analysis run --rm analysis
echo ""

echo "=========================================="
echo "✅ PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Results are in the 'results/' directory:"
echo "  - baseline_results_sample.json"
echo "  - improved_results_sample.json"
echo "  - cot_results_sample.json"
echo "  - opro_results_sample.json"
echo "  - comparison_sample.csv"
echo "  - analysis_report_sample.txt"
echo ""
echo "To run on FULL dataset (more expensive):"
echo "  1. Edit docker-compose.yml: change USE_SAMPLE=true to USE_SAMPLE=false"
echo "  2. Run: ./run_all.sh"
echo ""