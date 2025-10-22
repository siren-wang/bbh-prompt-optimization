"""
Baseline Evaluation Script with Proper Scoring Function
Tests basic prompting without improvements
"""

import json
import os
from pathlib import Path
import sys
from tqdm import tqdm

# Import our scoring module
sys.path.insert(0, os.path.dirname(__file__))
from scoring import BBHScorer

def load_task_data(task_name, use_sample=False):
    """Load full or sample data"""
    if use_sample:
        path = Path(f'data/sample/{task_name}_sample.json')
    else:
        path = Path(f'data/{task_name}.json')
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)

def create_base_prompt_template():
    """
    Create basic prompt template without improvements
    Template has {question} placeholder for the actual question
    """
    return "Q: {question}\nA:"

def evaluate_task(task_name, scorer, use_sample=False):
    """
    Evaluate a single task with base prompt using scoring function
    
    Args:
        task_name: Name of the BBH task
        scorer: BBHScorer instance
        use_sample: Whether to use sample or full dataset
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {task_name}")
    print(f"Mode: {'SAMPLE' if use_sample else 'FULL'}")
    print(f"{'='*60}")
    
    # Load data
    data = load_task_data(task_name, use_sample)
    examples = data['examples']
    
    # Create base prompt template
    prompt_template = create_base_prompt_template()
    
    print(f"Prompt Template: {prompt_template}")
    print(f"Number of examples: {len(examples)}")
    print()
    
    # Use scoring function to evaluate
    results = scorer.score_prompt(
        prompt_template=prompt_template,
        examples=examples,
        use_semantic=False,  # Use exact matching for baseline
        verbose=True,  # Show first 3 examples
        delay=0.1  # Rate limiting
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
    
    # Check if using sample mode (default for safety)
    USE_SAMPLE = os.getenv('USE_SAMPLE', 'true').lower() == 'true'
    
    # Selected tasks
    tasks = [
        'date_understanding',
        'logical_deduction_three_objects',
        'tracking_shuffled_objects_five_objects'
    ]
    
    print("\n" + "="*60)
    print("BASELINE EVALUATION WITH PROPER SCORING")
    print("="*60)
    print(f"Mode: {'SAMPLE (20 examples/task)' if USE_SAMPLE else 'FULL DATASET'}")
    print(f"Model: Gemini 1.5 Flash")
    print(f"Tasks: {len(tasks)}")
    print(f"Scoring: Exact match with #### prefix parsing")
    print("="*60)
    
    if not USE_SAMPLE:
        print("\n⚠️  WARNING: Running on FULL dataset!")
        print("This will use more API calls and cost more.")
        import time
        print("Press Ctrl+C to cancel, or wait 5 seconds to continue...")
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
    output_path = results_dir / f'baseline_results{suffix}.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - BASELINE RESULTS")
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
    
    # Estimate cost
    estimated_cost = total_examples * 0.0001  # Rough estimate for Gemini Flash
    print(f"💰 Estimated Cost: ${estimated_cost:.3f}")
    
    print(f"\n✅ Results saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()