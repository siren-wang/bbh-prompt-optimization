# BIG-Bench-Hard Prompt Engineering Assignment

A cost-efficient implementation of prompt engineering techniques on BIG-Bench-Hard tasks using Docker.

## 📋 Overview

This project evaluates different prompt engineering methods on BIG-Bench-Hard reasoning tasks:

1. **Baseline**: Simple prompting without improvements
2. **Improved Prompts**: Best practice techniques (role prompting, few-shot, clear instructions)
3. **Chain-of-Thought (CoT)**: Step-by-step reasoning prompts
4. **OPRO**: Automated prompt optimization using LLM as optimizer

**Model Used**: Google Gemini 1.5 Flash (most cost-efficient option)

## 🗂️ Project Structure

```
bbh-prompt-optimization/
├── data/                          # Your BBH task JSON files
│   └── sample/                    # Auto-generated sample datasets
├── src/
│   ├── data_preparation.py        # Creates sample datasets
│   ├── baseline_evaluation.py     # Baseline experiments
│   ├── improved_prompt_evaluation.py  # Improved prompting
│   ├── cot_evaluation.py          # Chain-of-Thought
│   ├── opro_optimization.py       # OPRO optimization
│   └── analyze_results.py         # Results analysis
├── results/                       # Generated results (JSON, CSV, reports)
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Multi-service orchestration
├── requirements.txt               # Python dependencies
├── .env.example                   # Example environment variables
├── run_all.sh                     # Run complete pipeline
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Google Gemini API key (get free at https://makersuite.google.com/app/apikey)
- BIG-Bench-Hard data files in `data/` folder

### Step 1: Setup

1. **Clone or create your repository**:
```bash
git clone <your-repo-url>
cd bbh-prompt-optimization
```

2. **Add BBH data files**:
   - Download JSON files from [BIG-Bench-Hard repo](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh)
   - Place them in the `data/` folder
   - Required files:
     - `date_understanding.json`
     - `logical_deduction_three_objects.json`
     - `tracking_shuffled_objects_five_objects.json`

3. **Create environment file**:
```bash
cp .env.example .env
```

4. **Add your API key to `.env`**:
```bash
# Edit .env and add your key
GOOGLE_API_KEY=your_actual_api_key_here
USE_SAMPLE=true
```

### Step 2: Run Complete Pipeline

**Option A: Run everything at once (Recommended)**
```bash
chmod +x run_all.sh
./run_all.sh
```

This will:
- Create sample datasets (20 examples each)
- Run baseline evaluation
- Run improved prompt evaluation
- Run Chain-of-Thought evaluation
- Run OPRO optimization
- Generate analysis and comparison

**Estimated time**: 20-30 minutes  
**Estimated cost**: ~$0.40

**Option B: Run steps individually**

```bash
# Build Docker image
docker-compose build

# Step 1: Create sample datasets
docker-compose --profile setup run --rm data-prep

# Step 2: Baseline
docker-compose --profile eval run --rm baseline

# Step 3: Improved prompts
docker-compose --profile eval run --rm improved

# Step 4: Chain-of-Thought
docker-compose --profile eval run --rm cot

# Step 5: OPRO optimization
docker-compose --profile opro run --rm opro

# Step 6: Analyze results
docker-compose --profile analysis run --rm analysis
```

### Step 3: View Results

Results are saved in the `results/` directory:

- `baseline_results_sample.json` - Baseline accuracy scores
- `improved_results_sample.json` - Improved prompt scores
- `cot_results_sample.json` - Chain-of-Thought scores
- `opro_results_sample.json` - OPRO scores with best prompts
- `comparison_sample.csv` - Comparison table
- `analysis_report_sample.txt` - Full analysis report

**View comparison table**:
```bash
cat results/comparison_sample.csv
```

**View full report**:
```bash
cat results/analysis_report_sample.txt
```

## 📊 Expected Results

Based on typical performance improvements:

| Method | Expected Accuracy | Improvement |
|--------|------------------|-------------|
| Baseline | 30-40% | - |
| Improved Prompts | 40-50% | +10-15% |
| Chain-of-Thought | 50-65% | +20-30% |
| OPRO | 55-70% | +25-35% |

*Actual results vary by task complexity*

## 🔧 Troubleshooting

### API Key Issues

**Error**: `Invalid API key`
```bash
# Check your .env file
cat .env

# Make sure you copied .env.example to .env
cp .env.example .env

# Add your key without quotes
GOOGLE_API_KEY=AIza...your_key_here
```

### Docker Issues

**Error**: `Cannot connect to Docker daemon`
```bash
# Start Docker Desktop
# Or on Linux:
sudo systemctl start docker
```

**Error**: `Permission denied`
```bash
# Make run_all.sh executable
chmod +x run_all.sh
```

### Data File Issues

**Error**: `File not found: data/task_name.json`
```bash
# Make sure you have the BBH JSON files in data/
ls data/

# Download from: https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh
# Place files directly in data/ folder (not in subdirectories)
```

### Rate Limit Issues

**Error**: `Rate limit exceeded`
```bash
# The code includes rate limiting (0.1s delay between calls)
# If you still hit limits, increase the delay in the Python files:
# Change: time.sleep(0.1)
# To: time.sleep(0.5)
```

## 📝 Understanding the Code

### Baseline (baseline_evaluation.py)
- Simple prompt: `Q: {question}\nA:`
- No special techniques
- Establishes performance floor

### Improved Prompts (improved_prompt_evaluation.py)
- Role prompting: "You are an expert reasoning assistant"
- Few-shot examples (3 examples)
- Clear output formatting instructions

### Chain-of-Thought (cot_evaluation.py)
- Adds "Let's think step by step"
- Includes reasoning examples
- Longer outputs to capture reasoning

### OPRO (opro_optimization.py)
- Uses LLM to generate better prompts
- Iterative optimization (5 iterations)
- Tests prompts on small samples
- Automatically discovers effective patterns

## 🎯 Assignment Deliverables Checklist

- [x] GitHub repository with clean structure
- [x] Dockerfile and docker-compose.yml
- [x] All source code with comments
- [x] Sample datasets for cost-efficient testing
- [x] Baseline evaluation implementation
- [x] Improved prompt implementation
- [x] Chain-of-Thought implementation
- [x] OPRO optimization implementation
- [x] Results comparison and analysis
- [x] Comprehensive README
- [x] Cost optimization strategies

## 📈 How to Present Results

### 1. Create Comparison Table
```bash
# Results are automatically generated
cat results/comparison_sample.csv
```

### 2. Show Improvement Analysis
```bash
cat results/analysis_report_sample.txt
```

### 3. Best Prompts Discovered
The OPRO results file contains the best prompts found:
```bash
cat results/opro_results_sample.json
```

### 4. Create Visualizations (Optional)

You can import the CSV into Excel/Google Sheets for charts, or add matplotlib to create plots:

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/comparison_sample.csv')
df.set_index('Task')[['BASELINE', 'IMPROVED', 'COT', 'OPRO']].plot(kind='bar')
plt.ylabel('Accuracy')
plt.title('Prompt Engineering Methods Comparison')
plt.tight_layout()
plt.savefig('results/comparison.png')
```

## 🧪 Advanced Usage

### Run Single Task

```bash
# Run specific task with OPRO
docker-compose run --rm -e OPRO_TASK=date_understanding opro
```

### Modify Number of Examples

Edit `src/data_preparation.py`:
```python
create_sample_dataset(task, sample_size=100)  # Increase from 20
```

### Use Different LLM

If you want to use OpenAI instead:

1. Update `requirements.txt`:
```
openai>=1.0.0
```

2. Modify Python files to use OpenAI client:
```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
```
