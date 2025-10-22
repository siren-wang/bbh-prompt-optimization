"""
Data Preparation Script for BBH Tasks
Creates sample datasets for cost-efficient prototyping
"""

import json
import random
from pathlib import Path

def create_sample_dataset(task_name, sample_size=20, seed=42):
    """
    Create small sample for prototyping
    
    Args:
        task_name: Name of the BBH task (without .json)
        sample_size: Number of examples to sample
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Read full dataset
    input_path = Path(f'data/{task_name}.json')
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    examples = data['examples']
    sample = random.sample(examples, min(sample_size, len(examples)))
    
    # Create sample directory if it doesn't exist
    sample_dir = Path('data/sample')
    sample_dir.mkdir(exist_ok=True)
    
    # Save sample
    sample_data = data.copy()
    sample_data['examples'] = sample
    
    output_path = sample_dir / f'{task_name}_sample.json'
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"✅ Created sample: {len(sample)} examples from {len(examples)} ({output_path})")

def main():
    """Create samples for selected tasks"""
    
    # Select 3 tasks with different difficulty levels
    tasks = [
        'date_understanding',           # Easy
        'logical_deduction_three_objects',  # Medium
        'tracking_shuffled_objects_five_objects'  # Hard
    ]
    
    print("Creating sample datasets for cost-efficient prototyping...")
    print("=" * 60)
    
    for task in tasks:
        create_sample_dataset(task, sample_size=3)
    
    print("=" * 60)
    print("✅ Sample datasets created successfully!")
    print("\n📊 Directory structure:")
    print("data/")
    print("├── [original task files]")
    print("└── sample/")
    for task in tasks:
        print(f"    └── {task}_sample.json (20 examples)")

if __name__ == "__main__":
    main()