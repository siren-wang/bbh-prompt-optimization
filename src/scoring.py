"""
Scoring Module for BBH Prompt Evaluation - OpenAI
"""

import os
import re
from openai import OpenAI
import time

class BBHScorer:
    """
    Scoring function for BIG-Bench-Hard tasks uing OpenAI
    """
    
    def __init__(self, model_name='gpt-4o-mini'):
        """Initialize scorer with OpenAI model"""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model_name = model_name
        self.scoring_client = self.client  # Same client for scoring
        
    def format_prompt_with_answer_prefix(self, prompt_text):
        """Add answer formatting instruction to prompt"""
        formatted_prompt = f"""{prompt_text}

IMPORTANT: Output your final answer at the end with the prefix "####". 
For example: #### ANSWER

Your response:"""
        return formatted_prompt
    
    def call_llm(self, prompt, max_tokens=200, temperature=0):
        """
        Call OpenAI LLM with the prompt
        Includes retry logic for network issues
        """
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=120  # Longer timeout for China networks
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"⚠️  API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"❌ Failed after {max_retries} attempts")
                    return ""
    
    def parse_answer(self, llm_output):
        """Parse the LLM output to extract the final answer"""
        if "####" in llm_output:
            parts = llm_output.split("####")
            if len(parts) > 1:
                answer = parts[-1].strip()
                return answer
        
        # Fallback: if no #### found, try to get last line
        lines = llm_output.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            for prefix in ["Answer:", "A:", "Final answer:", "The answer is"]:
                if last_line.startswith(prefix):
                    last_line = last_line[len(prefix):].strip()
            return last_line
        
        return llm_output.strip()
    
    def check_exact_match(self, parsed_answer, ground_truth):
        """Check if parsed answer exactly matches ground truth"""
        if not parsed_answer or not ground_truth:
            return False
        
        parsed_norm = parsed_answer.lower().strip()
        truth_norm = ground_truth.lower().strip()
        
        if parsed_norm == truth_norm:
            return True
        
        if truth_norm in parsed_norm:
            return True
        
        parsed_letter = self._extract_multiple_choice(parsed_norm)
        truth_letter = self._extract_multiple_choice(truth_norm)
        
        if parsed_letter and truth_letter:
            return parsed_letter == truth_letter
        
        return False
    
    def _extract_multiple_choice(self, text):
        """Extract multiple choice letter from text"""
        match = re.search(r'\(?([A-E])\)?', text.upper())
        if match:
            return match.group(1)
        return None
    
    def check_semantic_similarity(self, parsed_answer, ground_truth):
        """Use scoring LLM to check semantic similarity"""
        if self.check_exact_match(parsed_answer, ground_truth):
            return True
        
        scoring_prompt = f"""Compare these two answers and determine if they are semantically equivalent.

Answer 1: {parsed_answer}
Answer 2: {ground_truth}

Are these answers semantically similar? Respond with ONLY "YES" or "NO".

####"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": scoring_prompt}
                ],
                temperature=0,
                max_tokens=10,
                timeout=120
            )
            result = response.choices[0].message.content.strip().upper()
            return "YES" in result
            
        except Exception as e:
            print(f"Error in semantic similarity check: {e}")
            return False
    
    def score_single_example(self, prompt_template, example, use_semantic=False):
        """Score a single example"""
        question_prompt = prompt_template.format(question=example['input'])
        full_prompt = self.format_prompt_with_answer_prefix(question_prompt)
        
        llm_output = self.call_llm(full_prompt)
        parsed_answer = self.parse_answer(llm_output)
        
        if use_semantic:
            is_correct = self.check_semantic_similarity(parsed_answer, example['target'])
        else:
            is_correct = self.check_exact_match(parsed_answer, example['target'])
        
        return {
            'question': example['input'],
            'llm_output': llm_output,
            'parsed_answer': parsed_answer,
            'ground_truth': example['target'],
            'is_correct': is_correct
        }
    
    def score_prompt(self, prompt_template, examples, use_semantic=False, 
                    verbose=False, delay=0.5):
        """
        Score a prompt template on multiple examples
        Longer delay for China networks
        """
        results = []
        correct_count = 0
        
        for i, example in enumerate(examples):
            result = self.score_single_example(
                prompt_template, 
                example, 
                use_semantic=use_semantic
            )
            
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            if verbose and i < 3:
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {result['question'][:80]}...")
                print(f"LLM Output: {result['llm_output'][:100]}...")
                print(f"Parsed: {result['parsed_answer']}")
                print(f"Truth: {result['ground_truth']}")
                print(f"Correct: {result['is_correct']}")
            
            # Longer delay for China networks
            time.sleep(delay)
        
        accuracy = correct_count / len(examples) if examples else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': len(examples),
            'detailed_results': results
        }


def test_scorer():
    """Test the scoring function"""
    print("Testing BBH Scorer (OpenAI)...\n")
    
    scorer = BBHScorer()
    
    test_example = {
        'input': 'Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?',
        'target': '12/14/1937'
    }
    
    simple_prompt = "Q: {question}\nA:"
    
    print("=" * 60)
    print("Test: Simple Prompt with OpenAI")
    print("=" * 60)
    
    result = scorer.score_single_example(simple_prompt, test_example)
    
    print(f"Question: {result['question']}")
    print(f"\nLLM Output:\n{result['llm_output']}")
    print(f"\nParsed Answer: {result['parsed_answer']}")
    print(f"Ground Truth: {result['ground_truth']}")
    print(f"Correct: {result['is_correct']}")
    
    print("\n✅ Scorer test complete!")


if __name__ == "__main__":
    test_scorer()