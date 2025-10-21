"""
OPRO (Optimization by PROmpting) Implementation with Train/Eval Split
Automated prompt optimization using LLM as optimizer
Uses separate training and evaluation sets as per assignment requirements
"""

import json
import os
from pathlib import Path
import sys
import random
import time

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

def split_train_eval(examples, train_size=20, eval_size=30, seed=42):
    """
    Split dataset into training and evaluation sets
    Training set: Used for OPRO optimization
    Evaluation set: Used for final accuracy measurement
    
    Args:
        examples: All examples from dataset
        train_size: Number of examples for training (OPRO optimization)
        eval_size: Number of examples for evaluation (final accuracy)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Shuffle examples
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    # Split
    train_examples = shuffled[:train_size]
    eval_examples = shuffled[train_size:train_size + eval_size]
    
    return train_examples, eval_examples

class OPROOptimizer:
    """OPRO Optimizer using LLM to generate better prompts"""
    
    def __init__(self, task_name, train_examples, scorer):
        self.task_name = task_name
        self.train_examples = train_examples  # For optimization
        self.scorer = scorer
        self.prompt_history = []
    
    def generate_initial_prompts(self, n=4):
        """Generate initial set of prompts to test"""
        
        initial_prompts = [
            # Very simple baseline
            "Q: {question}\nA:",
            
            # With instruction
            "Answer the following question carefully.\n\nQ: {question}\nA:",
            
            # With role
            "You are an expert problem solver.\n\nQ: {question}\nA:",
            
            # With reasoning request
            "Think carefully about this question and provide the correct answer.\n\nQ: {question}\nA:"
        ]
        
        return initial_prompts[:n]
    
    def evaluate_prompt_on_training(self, prompt_template, sample_size=None):
        """
        Evaluate a prompt template on TRAINING set
        This is used during optimization iterations
        
        Args:
            prompt_template: Template with {question} placeholder
            sample_size: How many training examples to use (None = all)
        """
        
        # Use subset of training data for faster optimization
        if sample_size and sample_size < len(self.train_examples):
            examples_to_use = random.sample(self.train_examples, sample_size)
        else:
            examples_to_use = self.train_examples
        
        # Evaluate using scorer
        results = self.scorer.score_prompt(
            prompt_template=prompt_template,
            examples=examples_to_use,
            use_semantic=False,
            verbose=False,  # No verbose during optimization
            delay=0.1
        )
        
        return results['accuracy']
    
    def create_meta_prompt(self):
        """
        Create meta-prompt for optimizer LLM
        This prompt asks the LLM to generate better prompts based on history
        """
        
        # Sort by score and get top prompts
        sorted_history = sorted(
            self.prompt_history,
            key=lambda x: x['score'],
            reverse=True
        )[:5]  # Top 5
        
        meta_prompt = f"""Your task is to generate improved prompts for solving {self.task_name.replace('_', ' ')} reasoning tasks.

These prompts will be used to ask an AI model to solve problems. You need to create prompts that will help the AI give correct answers.

Here are previous prompts and their accuracy scores on training data:

"""
        
        for i, item in enumerate(sorted_history, 1):
            meta_prompt += f"{i}. Prompt:\n\"{item['prompt']}\"\n"
            meta_prompt += f"   Accuracy: {item['score']:.1%}\n\n"
        
        meta_prompt += """Based on the patterns above, generate 4 NEW and DIFFERENT improved prompts.

Guidelines for creating good prompts:
- Analyze what made high-scoring prompts successful
- Consider adding: clear instructions, expert role, reasoning guidance, output format
- Each prompt MUST include {{question}} placeholder (with double curly braces)
- Make prompts diverse - try different approaches
- Keep prompts concise but effective

Generate exactly 4 new prompt templates. Format each prompt like this:

PROMPT 1:
[your first prompt with {{question}} placeholder]

PROMPT 2:
[your second prompt with {{question}} placeholder]

PROMPT 3:
[your third prompt with {{question}} placeholder]

PROMPT 4:
[your fourth prompt with {{question}} placeholder]
"""
        
        return meta_prompt
    
    def generate_new_prompts(self, meta_prompt, n=4):
        """Use optimizer LLM to generate new prompts based on history"""
        
        try:
            # Use scorer's model for meta-optimization
            response = self.scorer.model.generate_content(
                meta_prompt,
                generation_config={
                    'temperature': 0.8,  # Higher temperature for diversity
                    'max_output_tokens': 800,
                }
            )
            
            text = response.text
            
            # Parse prompts from response
            prompts = []
            
            # Split by "PROMPT X:" markers
            import re
            sections = re.split(r'PROMPT \d+:', text)
            
            for section in sections[1:]:  # Skip first empty section
                section = section.strip()
                if section and '{question}' in section:
                    # Clean up the prompt
                    # Remove any trailing explanation
                    lines = section.split('\n')
                    prompt_lines = []
                    for line in lines:
                        if line.strip():
                            prompt_lines.append(line)
                        # Stop at next PROMPT marker or explanation
                        if 'PROMPT' in line.upper() or line.startswith('This prompt'):
                            break
                    
                    prompt = '\n'.join(prompt_lines).strip()
                    if '{question}' in prompt:
                        prompts.append(prompt)
            
            # If parsing failed or not enough prompts, generate fallbacks
            if len(prompts) < n:
                print(f"⚠️  Only parsed {len(prompts)} prompts, adding fallbacks")
                
                fallback_prompts = [
                    "You are an expert analyst. Carefully examine this question and provide the correct answer.\n\nQ: {question}\nA:",
                    "Let's solve this step by step. Read the question carefully and determine the right answer.\n\nQ: {question}\nA:",
                    "As an expert in reasoning, analyze this question thoroughly and give your answer.\n\nQ: {question}\nA:",
                    "Think through this problem carefully to find the correct solution.\n\nQ: {question}\nA:"
                ]
                
                while len(prompts) < n and fallback_prompts:
                    prompts.append(fallback_prompts.pop(0))
            
            return prompts[:n]
            
        except Exception as e:
            print(f"❌ Error generating prompts: {e}")
            # Return fallback prompts
            return [
                "You are an expert. Solve this:\n\nQ: {question}\nA:",
                "Think carefully:\n\nQ: {question}\nA:",
                "Q: {question}\n\nProvide the answer:\nA:",
                "Analyze carefully:\n\nQ: {question}\nA:"
            ][:n]
    
    def optimize(self, n_iterations=5, prompts_per_iteration=4, train_sample_size=15):
        """
        Run OPRO optimization loop on TRAINING set
        
        Args:
            n_iterations: Number of optimization iterations
            prompts_per_iteration: Number of prompts to test per iteration
            train_sample_size: Number of training examples per prompt evaluation
        """
        
        print(f"\n{'='*60}")
        print(f"OPRO OPTIMIZATION: {self.task_name}")
        print(f"{'='*60}")
        print(f"Training examples: {len(self.train_examples)}")
        print(f"Iterations: {n_iterations}")
        print(f"Prompts per iteration: {prompts_per_iteration}")
        print(f"Training sample per prompt: {train_sample_size}")
        print(f"{'='*60}\n")
        
        # Start with initial prompts
        current_prompts = self.generate_initial_prompts(prompts_per_iteration)
        
        for iteration in range(n_iterations):
            print(f"\n{'--- Iteration %d/%d ---' % (iteration + 1, n_iterations)}")
            
            # Evaluate current prompts on TRAINING set
            for i, prompt in enumerate(current_prompts):
                print(f"\nTesting prompt {i+1}/{len(current_prompts)}...")
                print(f"Prompt preview: {prompt[:100]}...")
                
                score = self.evaluate_prompt_on_training(
                    prompt, 
                    sample_size=train_sample_size
                )
                
                self.prompt_history.append({
                    'prompt': prompt,
                    'score': score,
                    'iteration': iteration
                })
                
                print(f"✓ Training Accuracy: {score:.1%}")
            
            # Generate new prompts for next iteration (except last)
            if iteration < n_iterations - 1:
                print(f"\n🔄 Generating new prompts for iteration {iteration + 2}...")
                meta_prompt = self.create_meta_prompt()
                current_prompts = self.generate_new_prompts(
                    meta_prompt,
                    prompts_per_iteration
                )
        
        # Find best prompt from optimization
        best = max(self.prompt_history, key=lambda x: x['score'])
        
        print(f"\n{'='*60}")
        print("BEST PROMPT FROM OPTIMIZATION (on training set)")
        print(f"{'='*60}")
        print(f"Training Accuracy: {best['score']:.2%}")
        print(f"Iteration: {best['iteration'] + 1}")
        print(f"\nPrompt:\n{best['prompt']}")
        print(f"{'='*60}")
        
        return best

def evaluate_on_eval_set(task_name, best_prompt_template, eval_examples, scorer):
    """
    Evaluate the best prompt on EVALUATION set (separate from training)
    This gives the final accuracy score for the assignment
    """
    
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION ON EVALUATION SET")
    print(f"{'='*60}")
    print(f"Task: {task_name}")
    print(f"Evaluation examples: {len(eval_examples)}")
    print(f"{'='*60}\n")
    
    # Evaluate using scorer on eval set
    results = scorer.score_prompt(
        prompt_template=best_prompt_template,
        examples=eval_examples,
        use_semantic=False,
        verbose=True,  # Show first 3 examples
        delay=0.1
    )
    
    print(f"\n📊 Final Results on Evaluation Set:")
    print(f"   Correct: {results['correct']}/{results['total']}")
    print(f"   Accuracy: {results['accuracy']:.2%}")
    
    return {
        'accuracy': results['accuracy'],
        'correct': results['correct'],
        'total': results['total']
    }

def main():
    """Main OPRO optimization function"""
    
    USE_SAMPLE = os.getenv('USE_SAMPLE', 'true').lower() == 'true'
    
    tasks = [
        'date_understanding',
        'logical_deduction_three_objects',
        'tracking_shuffled_objects_five_objects'
    ]
    
    print("\n" + "="*60)
    print("OPRO OPTIMIZATION WITH TRAIN/EVAL SPLIT")
    print("="*60)
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Mode: {'SAMPLE' if USE_SAMPLE else 'FULL'}")
    print(f"Training: Used for prompt optimization")
    print(f"Evaluation: Used for final accuracy measurement")
    print("="*60)
    
    # Initialize scorer
    print("\n🔧 Initializing BBH Scorer...")
    scorer = BBHScorer()
    
    all_results = {}
    
    for task_name in tasks:
        print(f"\n\n{'#'*60}")
        print(f"# TASK: {task_name}")
        print(f"{'#'*60}")
        
        # Load data
        data = load_task_data(task_name, use_sample=USE_SAMPLE)
        examples = data['examples']
        
        # Split into train and eval
        train_examples, eval_examples = split_train_eval(
            examples,
            train_size=20,  # 20 for training (OPRO optimization)
            eval_size=30,   # 30 for evaluation (final accuracy)
            seed=42
        )
        
        print(f"\n📊 Data Split:")
        print(f"  Total examples: {len(examples)}")
        print(f"  Training set: {len(train_examples)} (for OPRO)")
        print(f"  Evaluation set: {len(eval_examples)} (for final accuracy)")
        
        # Run OPRO optimization on TRAINING set
        optimizer = OPROOptimizer(task_name, train_examples, scorer)
        best = optimizer.optimize(
            n_iterations=5,
            prompts_per_iteration=4,
            train_sample_size=15  # Use 15 of 20 training examples per prompt
        )
        
        # Evaluate best prompt on EVALUATION set
        final_result = evaluate_on_eval_set(
            task_name,
            best['prompt'],
            eval_examples,
            scorer
        )
        
        all_results[task_name] = {
            'best_prompt': best['prompt'],
            'training_accuracy': best['score'],
            'final_accuracy': final_result['accuracy'],
            'correct': final_result['correct'],
            'total': final_result['total']
        }
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    suffix = '_sample' if USE_SAMPLE else '_full'
    output_path = results_dir / f'opro_results{suffix}.json'
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - OPRO RESULTS")
    print("="*60)
    
    for task, result in all_results.items():
        print(f"\n{task}:")
        print(f"  Training Accuracy: {result['training_accuracy']:.2%}")
        print(f"  Final Eval Accuracy: {result['final_accuracy']:.2%}")
        print(f"  Eval Results: {result['correct']}/{result['total']}")
        print(f"  Best Prompt: {result['best_prompt'][:80]}...")
    
    # Calculate overall
    total_correct = sum(r['correct'] for r in all_results.values())
    total_examples = sum(r['total'] for r in all_results.values())
    overall = total_correct / total_examples if total_examples > 0 else 0
    
    print(f"\n📊 Overall Evaluation Accuracy: {overall:.2%} ({total_correct}/{total_examples})")
    print(f"\n✅ Results saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()