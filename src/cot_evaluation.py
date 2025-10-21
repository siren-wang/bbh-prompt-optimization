"""
Chain-of-Thought (CoT) Evaluation Script with Proper Scoring
"""

import json
import os
from pathlib import Path
import sys

# Import our scoring module
sys.path.insert(0, os.path.dirname(__file__))
from scoring import BBHScorer

def load_task_data(task_name, use_sample=False):
    """Load full or sample data"""
    if use_sample:
        path = Path(f'data/sample/{task_name}_sample.json')
    else:
        path = Path(f'data/{task_name}.json')
    
    with open(path, 'r') as f:
        return json.load(f)

def create_cot_prompt_template(task_name):
    """
    Create PURE Chain-of-Thought prompt template
    ONLY uses "Let's think step by step" - NO few-shot examples
    
    Returns template with {question} placeholder
    """
    
    # Task-specific CoT prompts with step-by-step guidance
    task_templates = {
        'date_understanding': """You are an expert at date calculations.

Q: {question}

Let's think step by step to find the answer:
1. First, identify the reference date mentioned
2. Then, determine what operation is needed (add/subtract days)
3. Perform the calculation carefully
4. Format the final answer as MM/DD/YYYY

A:""",
        
        'logical_deduction_three_objects': """You are an expert in logical reasoning.

Q: {question}

Let's think step by step to solve this:
1. First, list all the objects mentioned
2. Then, identify each relationship statement (e.g., "X is above Y")
3. Build the complete ordering by combining all relationships
4. Finally, determine which option correctly describes the arrangement

A:""",
        
        'tracking_shuffled_objects_five_objects': """You are an expert at tracking object movements.

Q: {question}

Let's think step by step to track the objects:
1. First, note the initial state (who has what)
2. Then, process each swap operation one by one
3. Update positions after each swap
4. Finally, determine the final position of the requested object

A:"""
    }
    
    # Get task-specific template or use generic CoT
    if task_name in task_templates:
        return task_templates[task_name]
    else:
        # Generic pure CoT template
        return """You are an expert problem solver.

Q: {question}

Let's think step by step to solve this problem:

A:"""

def evaluate_task(task_name, scorer, use_sample=False):
    """Evaluate a single task with pure CoT prompting (no few-shot)"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {task_name}")
    print(f"Mode: {'SAMPLE' if use_sample else 'FULL'}")
    print(f"{'='*60}")
    
    # Load data
    data = load_task_data(task_name, use_sample)
    examples = data['examples']
    
    # Create pure CoT prompt template
    prompt_template = create_cot_prompt_template(task_name)
    
    print(f"Prompt Template:")
    print(prompt_template)
    print(f"\nKey technique: 'Let's think step by step'")
    print(f"Number of test examples: {len(examples)}")
    print()
    
    # Use scoring function to evaluate
    results = scorer.score_prompt(
        prompt_template=prompt_template,
        examples=examples,
        use_semantic=False,  # Exact matching
        verbose=True,  # Show first 3 examples
        delay=0.1
    )
    
    print(f"\n📊 Results for {task_name}:")
    print(f"   Correct: {results['correct']}/{results['total']}")
    print(f"   Accuracy: {results['accuracy']:.2%}")
    
    return {
        'accuracy': results['accuracy'],
        'correct': results['correct'],
        'total': results['total'],
        'prompt_template': prompt_template
    }

def main():
    """Main evaluation function"""
    
    USE_SAMPLE = os.getenv('USE_SAMPLE', 'true').lower() == 'true'
    
    tasks = [
        'date_understanding',
        'logical_deduction_three_objects',
        'tracking_shuffled_objects_five_objects'
    ]
    
    print("\n" + "="*60)
    print("CHAIN-OF-THOUGHT EVALUATION (PURE CoT)")
    print("="*60)
    print(f"Mode: {'SAMPLE' if USE_SAMPLE else 'FULL DATASET'}")
    print(f"Model: Gemini 1.5 Flash")
    print(f"Technique: Chain-of-Thought prompting")
    print(f"  ✓ Uses 'Let's think step by step'")
    print(f"  ✓ Provides step-by-step reasoning structure")
    print(f"  ✗ NO few-shot examples")
    print(f"  ✗ NO Tree-of-Thoughts")
    print(f"  ✗ NO Meta prompting")
    print(f"  ✗ NO Self-consistency")
    print(f"Scoring: Exact match with #### prefix parsing")
    print("="*60)
    
    if not USE_SAMPLE:
        print("\n⚠️  WARNING: Running on FULL dataset!")
        import time
        time.sleep(5)
    
    # Initialize scorer
    print("\n🔧 Initializing BBH Scorer...")
    scorer = BBHScorer()
    
    results = {}
    
    for task in tasks:
        try:
            results[task] = evaluate_task(task, scorer, USE_SAMPLE)
        except Exception as e:
            print(f"\n❌ Failed to evaluate {task}: {e}")
            results[task] = {
                'accuracy': 0,
                'correct': 0,
                'total': 0,
                'error_message': str(e)
            }
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    suffix = '_sample' if USE_SAMPLE else '_full'
    output_path = results_dir / f'cot_results{suffix}.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - CHAIN-OF-THOUGHT RESULTS")
    print("="*60)
    
    total_correct = 0
    total_examples = 0
    
    for task, result in results.items():
        print(f"{task}:")
        print(f"  Accuracy: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
        total_correct += result['correct']
        total_examples += result['total']
    
    overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
    print(f"\n📊 Overall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_examples})")
    
    estimated_cost = total_examples * 0.00015  # Slightly higher due to longer outputs
    print(f"💰 Estimated Cost: ${estimated_cost:.3f}")
    
    print(f"\n✅ Results saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()