# BIG-Bench-Hard Prompt Engineering Assignment

A systematic evaluation of prompt engineering techniques on BIG-Bench-Hard reasoning tasks.

---

## 📊 Results

### Performance Comparison

Experiments conducted with 50 examples per task (3 tasks total = 150 examples):

| Method | Date Understanding | Logical Deduction | Object Tracking | **Overall** |
|--------|-------------------|-------------------|-----------------|-------------|
| **Baseline** | 15% | 85% | 15% | **38%** |
| **Improved Prompts** | 5% | 70% | 10% | **28%** |
| **Chain-of-Thought** | 0% | 100% | 100% | **67%** |
| **OPRO** | 45% | 90% | 80% | **72%** |

---

## 🔬 Project Overview

This project implements and compares four prompt engineering approaches on BIG-Bench-Hard tasks:

1. **Baseline**: Minimal prompting (Q: {question}\nA:)
2. **Improved Prompts**: Manual engineering with role prompting and instructions
3. **Chain-of-Thought**: Pure CoT with "Let's think step by step"
4. **OPRO**: Automated prompt optimization with separate train/eval sets

### Tasks Evaluated

- **Date Understanding**: Temporal reasoning and date calculations
- **Logical Deduction (3 Objects)**: Ordering and relationship reasoning
- **Tracking Shuffled Objects (5 Objects)**: State tracking through operations

### Model Used

- **Gemini 1.5 Flash** (most cost-efficient option)
- Alternative: Can be adapted for OpenAI GPT-4o-mini

---

## How to Reproduce
Prerequisites

Python 3.9+
Google Gemini API key

### Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/bbh-prompt-optimization.git
cd bbh-prompt-optimization

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Key

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
# GOOGLE_API_KEY=your_key_here
```

### Step 3: Prepare Data

Download BIG-Bench-Hard data files and place in `data/` folder:
- `date_understanding.json`
- `logical_deduction_three_objects.json`
- `tracking_shuffled_objects_five_objects.json`

Source: https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh

```bash
# Create sample datasets (50 examples each)
python3 src/data_preparation.py
```

### Step 4: Run Experiments

**Option A: Run Complete Pipeline**

```bash
# If in China, use proxychains4
proxychains4 python3 src/baseline_evaluation.py
proxychains4 python3 src/improved_prompt_evaluation.py
proxychains4 python3 src/cot_evaluation.py
proxychains4 python3 src/opro_optimization.py

# Analyze results
python3 src/analyze_results.py
```

**Option B: Run Individual Experiments**

```bash
# Baseline
python3 src/baseline_evaluation.py

# Improved prompts
python3 src/improved_prompt_evaluation.py

# Chain-of-Thought
python3 src/cot_evaluation.py

# OPRO (takes longer)
python3 src/opro_optimization.py

# Analysis
echo "sample" | python3 src/analyze_results.py
```

### Step 5: View Results

Results are saved in the `results/` directory:

```bash
# View comparison table
cat results/comparison_sample.csv

# View full analysis
cat results/analysis_report_sample.txt

# View individual results
cat results/baseline_results_sample.json
cat results/improved_results_sample.json
cat results/cot_results_sample.json
cat results/opro_results_sample.json
```

---

## 📁 Project Structure

```
bbh-prompt-optimization/
├── data/                          # BBH task data files
│   ├── date_understanding.json
│   ├── logical_deduction_three_objects.json
│   ├── tracking_shuffled_objects_five_objects.json
│   └── sample/                    # Auto-generated samples
├── src/
│   ├── scoring.py                 # Scoring function with #### parsing
│   ├── data_preparation.py        # Sample dataset creation
│   ├── baseline_evaluation.py     # Baseline experiments
│   ├── improved_prompt_evaluation.py  # Manual prompt engineering
│   ├── cot_evaluation.py          # Chain-of-Thought
│   ├── opro_optimization.py       # Automated optimization
│   └── analyze_results.py         # Results comparison
├── results/                       # Generated results
│   ├── baseline_results_sample.json
│   ├── improved_results_sample.json
│   ├── cot_results_sample.json
│   ├── opro_results_sample.json
│   ├── comparison_sample.csv
│   └── analysis_report_sample.txt
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🔑 Key Implementation Details

### Scoring Function (`scoring.py`)

Implements proper answer parsing as per assignment requirements:

1. **Format prompt** with `####` prefix instruction
2. **Call LLM** to get response
3. **Parse answer** from text after `####`
4. **Check correctness** using exact match or semantic similarity

```python
# Example usage
scorer = BBHScorer()
result = scorer.score_single_example(prompt_template, example)
# Returns: {'is_correct': True/False, 'parsed_answer': '(A)', ...}
```

### Improved Prompts (Manual Engineering)

**Techniques used:**
- ✅ Role prompting: "You are an expert problem solver"
- ✅ Detailed instructions: Step-by-step guidance
- ✅ Reasoning request: "Think carefully..."
- ❌ NO few-shot examples (as per requirements)

### Chain-of-Thought

**Pure CoT only:**
- ✅ "Let's think step by step" instruction
- ✅ Step-by-step reasoning structure
- ❌ NO few-shot examples
- ❌ NO Tree-of-Thoughts, Meta prompting, or Self-consistency

### OPRO (Automated Optimization)

**Implementation:**
- ✅ Separate training (20 examples) and evaluation (30 examples) sets
- ✅ Training set used ONLY for prompt optimization
- ✅ Evaluation set used ONLY for final accuracy
- ✅ LLM generates new prompts based on performance history
- ✅ Iterative improvement over 5 iterations
