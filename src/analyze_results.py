"""
Results Analysis and Visualization
Compares all prompt engineering methods
"""

import json
from pathlib import Path
import pandas as pd

def load_results(use_sample=False):
    """Load all result files"""
    
    suffix = '_sample' if use_sample else '_full'
    results_dir = Path('results')
    
    results = {}
    
    methods = ['baseline', 'improved', 'cot', 'opro']
    
    for method in methods:
        file_path = results_dir / f'{method}_results{suffix}.json'
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                results[method] = json.load(f)
        else:
            print(f"⚠️  {file_path} not found, skipping {method}")
    
    return results

def create_comparison_table(results):
    """Create comparison DataFrame"""
    
    # Get all tasks
    tasks = set()
    for method_results in results.values():
        tasks.update(method_results.keys())
    
    tasks = sorted(list(tasks))
    
    # Build comparison data
    comparison_data = []
    
    for task in tasks:
        row = {'Task': task.replace('_', ' ').title()}
        
        for method in ['baseline', 'improved', 'cot', 'opro']:
            if method in results and task in results[method]:
                if method == 'opro':
                    # OPRO has different structure
                    acc = results[method][task].get('final_accuracy', 0)
                else:
                    acc = results[method][task].get('accuracy', 0)
                row[method.upper()] = f"{acc:.2%}"
            else:
                row[method.upper()] = "N/A"
        
        comparison_data.append(row)
    
    # Add overall row
    overall_row = {'Task': 'OVERALL'}
    
    for method in ['baseline', 'improved', 'cot', 'opro']:
        if method not in results:
            overall_row[method.upper()] = "N/A"
            continue
        
        total_correct = 0
        total_examples = 0
        
        for task in results[method]:
            if method == 'opro':
                total_correct += results[method][task].get('correct', 0)
                total_examples += results[method][task].get('total', 0)
            else:
                total_correct += results[method][task].get('correct', 0)
                total_examples += results[method][task].get('total', 0)
        
        if total_examples > 0:
            overall_acc = total_correct / total_examples
            overall_row[method.upper()] = f"{overall_acc:.2%}"
        else:
            overall_row[method.upper()] = "N/A"
    
    comparison_data.append(overall_row)
    
    df = pd.DataFrame(comparison_data)
    return df

def calculate_improvements(results):
    """Calculate improvement percentages"""
    
    if 'baseline' not in results:
        print("⚠️  Baseline results not found, cannot calculate improvements")
        return None
    
    improvements = {}
    
    # Get baseline overall accuracy
    baseline_correct = 0
    baseline_total = 0
    
    for task, result in results['baseline'].items():
        baseline_correct += result.get('correct', 0)
        baseline_total += result.get('total', 0)
    
    baseline_acc = baseline_correct / baseline_total if baseline_total > 0 else 0
    
    print(f"\n{'='*60}")
    print("IMPROVEMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"Baseline Accuracy: {baseline_acc:.2%}")
    print(f"{'='*60}\n")
    
    for method in ['improved', 'cot', 'opro']:
        if method not in results:
            continue
        
        method_correct = 0
        method_total = 0
        
        for task, result in results[method].items():
            if method == 'opro':
                method_correct += result.get('correct', 0)
                method_total += result.get('total', 0)
            else:
                method_correct += result.get('correct', 0)
                method_total += result.get('total', 0)
        
        method_acc = method_correct / method_total if method_total > 0 else 0
        
        absolute_improvement = method_acc - baseline_acc
        relative_improvement = (absolute_improvement / baseline_acc * 100) if baseline_acc > 0 else 0
        
        improvements[method] = {
            'accuracy': method_acc,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement
        }
        
        print(f"{method.upper()}:")
        print(f"  Accuracy: {method_acc:.2%}")
        print(f"  Absolute Improvement: {absolute_improvement:+.2%}")
        print(f"  Relative Improvement: {relative_improvement:+.1f}%")
        print()
    
    return improvements

def print_best_prompts(results):
    """Print best prompts from OPRO"""
    
    if 'opro' not in results:
        return
    
    print(f"\n{'='*60}")
    print("BEST PROMPTS DISCOVERED BY OPRO")
    print(f"{'='*60}\n")
    
    for task, result in results['opro'].items():
        print(f"Task: {task}")
        print(f"Accuracy: {result.get('final_accuracy', 0):.2%}")
        print(f"Prompt: {result.get('best_prompt', 'N/A')}")
        print("-" * 60)
        print()

def main():
    """Main analysis function"""
    
    USE_SAMPLE = input("Analyze sample or full results? (sample/full) [sample]: ").strip().lower()
    if USE_SAMPLE == '':
        USE_SAMPLE = 'sample'
    
    use_sample = USE_SAMPLE == 'sample'
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {'SAMPLE' if use_sample else 'FULL'} RESULTS")
    print(f"{'='*60}\n")
    
    # Load all results
    results = load_results(use_sample)
    
    if not results:
        print("❌ No results found. Please run experiments first.")
        return
    
    print(f"Found results for: {', '.join(results.keys())}\n")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save to CSV
    results_dir = Path('results')
    suffix = '_sample' if use_sample else '_full'
    csv_path = results_dir / f'comparison{suffix}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Table saved to: {csv_path}")
    
    # Calculate improvements
    improvements = calculate_improvements(results)
    
    # Print best prompts
    print_best_prompts(results)
    
    # Create summary report
    report_path = results_dir / f'analysis_report{suffix}.txt'
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BIG-BENCH-HARD PROMPT ENGINEERING ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Dataset: {'Sample (20 examples/task)' if use_sample else 'Full Dataset'}\n")
        f.write(f"Model: Gemini 1.5 Flash\n\n")
        
        f.write("COMPARISON TABLE\n")
        f.write("-"*60 + "\n")
        f.write(df.to_string(index=False) + "\n\n")
        
        if improvements:
            f.write("IMPROVEMENTS OVER BASELINE\n")
            f.write("-"*60 + "\n")
            for method, stats in improvements.items():
                f.write(f"\n{method.upper()}:\n")
                f.write(f"  Accuracy: {stats['accuracy']:.2%}\n")
                f.write(f"  Absolute Improvement: {stats['absolute_improvement']:+.2%}\n")
                f.write(f"  Relative Improvement: {stats['relative_improvement']:+.1f}%\n")
        
        if 'opro' in results:
            f.write("\n\nBEST PROMPTS FROM OPRO\n")
            f.write("-"*60 + "\n")
            for task, result in results['opro'].items():
                f.write(f"\n{task}:\n")
                f.write(f"  Accuracy: {result.get('final_accuracy', 0):.2%}\n")
                f.write(f"  Prompt: {result.get('best_prompt', 'N/A')}\n")
    
    print(f"\n✅ Full report saved to: {report_path}")
    print("="*60)

if __name__ == "__main__":
    main()