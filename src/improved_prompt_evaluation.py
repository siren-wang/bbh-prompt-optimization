"""
Improved Prompt Evaluation Script with Proper Scoring
Applies best practice prompt engineering techniques WITHOUT few-shot examples
Improvements: Role prompting, detailed instructions, reasoning guidance
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
    
    with open(path, 'r') as f:
        return json.load(f)

def create_improved_prompt_template(task_name):
    """
    Create improved prompt template with best practices:
    - Role prompting (expert persona)
    - Detailed instructions
    - Reasoning guidance (think through the question)
    
    NO FEW-SHOT EXAMPLES
    
    Returns template with {question} placeholder
    """
    
    # Task-specific role and instructions
    task_templates = {
        'date_understanding': """You are an expert at date calculations and temporal reasoning.

Your task: Carefully analyze the given date-related question and compute the correct answer.

Instruction: Read the question carefully, determine what to do, think through the problem, perform the calculation accurately, choose the correct option

Q: {question}

Think through this carefully and provide the correct date.
A:""",
        
        'logical_deduction_three_objects': """You are an expert in logical reasoning and deductive analysis.

Your task: Analyze the given statements about objects and their relationships, then determine the correct logical conclusion.

Instruction: Read the question carefully, determine what to do, think through the problem, perform the calculation accurately, choose the correct option

Q: {question}

Think through the logical relationships and select the correct answer.
A:""",
        
        'tracking_shuffled_objects_five_objects': """You are an expert at tracking object positions through sequences of operations.

Your task: Track how objects change positions through a series of swaps or movements.

Instruction: Read the question carefully, determine what to do, think through the problem, perform the calculation accurately, choose the correct option

Q: {question}

Carefully track each operation and determine the final answer.
A:"""
    }
    
    # Get task-specific template or use generic
    if task_name in task_templates:
        return task_templates[task_name]
    else:
        # Generic improved template
        return """You are an expert problem-solving assistant with strong analytical skills.

Your task: Carefully analyze the given question and provide an accurate, well-reasoned answer.

Instruction: Read the question carefully, determine what to do, think through the problem, perform the calculation accurately, choose the correct option
Q: {question}

Reason through this problem carefully and provide your answer.
A:"""

def evaluate_task(task_name, scorer, use_sample=False):
    """Evaluate a single task with improved prompt (no few-shot)"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {task_name}")
    print(f"Mode: {'SAMPLE' if use_sample else 'FULL'}")
    print(f"{'='*60}")
    
    # Load data
    data = load_task_data(task_name, use_sample)
    examples = data['examples']
    
    # Create improved prompt template (no few-shot)
    prompt_template = create_improved_prompt_template(task_name)
    
    print(f"Prompt Template (first 300 chars):")
    print(prompt_template[:300] + "...")
    print(f"\nNumber of test examples: {len(examples)}")
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
        'prompt_template': prompt_template[:500]  # Save first 500 chars
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
    print("IMPROVED PROMPT EVALUATION (NO FEW-SHOT)")
    print("="*60)
    print(f"Mode: {'SAMPLE' if USE_SAMPLE else 'FULL DATASET'}")
    print(f"Model: Gemini 1.5 Flash")
    print(f"Techniques:")
    print(f"  ✓ Role prompting (expert persona)")
    print(f"  ✓ Detailed step-by-step instructions")
    print(f"  ✓ Reasoning guidance (think through the question)")
    print(f"  ✗ NO few-shot examples")
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
    output_path = results_dir / f'improved_results{suffix}.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - IMPROVED PROMPT RESULTS (NO FEW-SHOT)")
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
    
    estimated_cost = total_examples * 0.00012
    print(f"💰 Estimated Cost: ${estimated_cost:.3f}")
    
    print(f"\n✅ Results saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()