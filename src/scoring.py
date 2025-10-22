"""
Scoring Module for BBH Prompt Evaluation
Implements proper answer parsing and semantic similarity checking
Fixed for China network issues
"""

import os
import re
import google.generativeai as genai
import time

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class BBHScorer:
    """
    Scoring function for BIG-Bench-Hard tasks
    - Takes a prompt and runs it through LLM
    - Parses output for the answer
    - Checks if parsed answer is correct
    """
    
    def __init__(self, model_name='gemini-2.0-flash'):
        """Initialize scorer with LLM model - using Flash 8B (cheapest)"""
        # gemini-2.0-flash is the cheapest model
        try:
            self.model = genai.GenerativeModel(model_name)
            self.scoring_llm = genai.GenerativeModel(model_name)
            print(f"✓ Using model: {model_name}")
        except Exception as e:
            print(f"⚠️  Model {model_name} not available: {str(e)[:100]}")
            print("Trying alternative models...")
            # Try alternative models in order of cost (cheapest first)
            for alt_model in ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-2.0-flash-exp', 'gemini-pro']:
                try:
                    print(f"  Trying: {alt_model}")
                    self.model = genai.GenerativeModel(alt_model)
                    self.scoring_llm = genai.GenerativeModel(alt_model)
                    print(f"  ✓ Using model: {alt_model}")
                    break
                except Exception as e2:
                    print(f"  ✗ Failed: {str(e2)[:20]}")
                    continue
        
    def format_prompt_with_answer_prefix(self, prompt_text):
        """
        Add answer formatting instruction to prompt
        This ensures consistent parsing of LLM output
        """
        formatted_prompt = f"""{prompt_text}

IMPORTANT: Output your final answer at the end with the prefix "####". 
For example: #### ANSWER

Your response:"""
        return formatted_prompt
    
    def call_llm(self, prompt, max_tokens=500, temperature=0):
        """
        Call LLM with the prompt with retry logic
        Fixed response handling for complex responses
        
        Args:
            prompt: The formatted prompt
            max_tokens: Maximum output tokens
            temperature: Sampling temperature (0 for deterministic)
        
        Returns:
            LLM response text
        """
        max_retries = 3
        retry_delay = 3
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                )
                
                # Handle response safely
                try:
                    # Try simple accessor first
                    return response.text.strip()
                except ValueError:
                    # If that fails, extract from candidates
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            text_parts = []
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    text_parts.append(part.text)
                            return ' '.join(text_parts).strip()
                    return ""
                
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️  API call failed (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}")
                
                # Check if it's a blocking/503 error (China firewall)
                if '503' in error_msg or 'failed to connect' in error_msg.lower() or 'timeout' in error_msg.lower():
                    print("❌ Google API blocked or timing out (likely Great Firewall)")
                    if attempt >= max_retries - 1:
                        print("\n" + "="*60)
                        print("⚠️  CRITICAL: Google API not accessible from China")
                        print("="*60)
                        print("RECOMMENDED: Switch to OpenAI (works in China)")
                        print("1. pip install openai")
                        print("2. Get key: https://platform.openai.com/api-keys")
                        print("3. Update .env: OPENAI_API_KEY=sk-...")
                        print("="*60)
                        return ""
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"❌ Failed after {max_retries} attempts")
                    return ""
        
        return ""
    
    def parse_answer(self, llm_output):
        """
        Parse the LLM output to extract the final answer
        Looks for the "####" prefix
        
        Args:
            llm_output: Raw LLM response
        
        Returns:
            Parsed answer string, or None if parsing fails
        """
        if not llm_output:
            return ""
            
        # Look for #### prefix
        if "####" in llm_output:
            # Extract everything after ####
            parts = llm_output.split("####")
            if len(parts) > 1:
                answer = parts[-1].strip()
                return answer
        
        # Fallback: if no #### found, try to get last line
        lines = llm_output.strip().split('\n')
        if lines:
            # Check if last line looks like an answer
            last_line = lines[-1].strip()
            # Remove common prefixes
            for prefix in ["Answer:", "A:", "Final answer:", "The answer is"]:
                if last_line.startswith(prefix):
                    last_line = last_line[len(prefix):].strip()
            return last_line
        
        return llm_output.strip()
    
    def check_exact_match(self, parsed_answer, ground_truth):
        """
        Check if parsed answer exactly matches ground truth
        
        Args:
            parsed_answer: Answer extracted from LLM output
            ground_truth: Correct answer
        
        Returns:
            Boolean indicating if answers match
        """
        if not parsed_answer or not ground_truth:
            return False
        
        # Normalize both answers
        parsed_norm = parsed_answer.lower().strip()
        truth_norm = ground_truth.lower().strip()
        
        # Direct match
        if parsed_norm == truth_norm:
            return True
        
        # Check if ground truth is contained in parsed answer
        if truth_norm in parsed_norm:
            return True
        
        # For multiple choice answers like "(A)", "(B)", etc.
        # Extract just the letter
        parsed_letter = self._extract_multiple_choice(parsed_norm)
        truth_letter = self._extract_multiple_choice(truth_norm)
        
        if parsed_letter and truth_letter:
            return parsed_letter == truth_letter
        
        return False
    
    def _extract_multiple_choice(self, text):
        """Extract multiple choice letter from text like '(A)' or 'A'"""
        # Match patterns like (A), (B), or just A, B
        match = re.search(r'\(?([A-E])\)?', text.upper())
        if match:
            return match.group(1)
        return None
    
    def check_semantic_similarity(self, parsed_answer, ground_truth):
        """
        Use a scoring LLM to check semantic similarity
        for free-form text answers
        
        Args:
            parsed_answer: Answer from model
            ground_truth: Correct answer
        
        Returns:
            Boolean indicating semantic similarity
        """
        # First try exact match
        if self.check_exact_match(parsed_answer, ground_truth):
            return True
        
        # Use LLM to judge semantic similarity
        scoring_prompt = f"""Compare these two answers and determine if they are semantically equivalent or convey the same meaning.

Answer 1: {parsed_answer}
Answer 2: {ground_truth}

Are these answers semantically similar? Consider them similar if they express the same core idea, even if worded differently.

Respond with ONLY "YES" or "NO".

####"""
        
        try:
            response = self.scoring_llm.generate_content(
                scoring_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=10,
                )
            )
            
            # Handle response safely
            try:
                result = response.text.strip().upper()
            except ValueError:
                # Extract from candidates if simple accessor fails
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        result = candidate.content.parts[0].text.strip().upper()
                    else:
                        return False
                else:
                    return False
            
            return "YES" in result
            
        except Exception as e:
            print(f"Error in semantic similarity check: {e}")
            # Fallback to exact match
            return False
    
    def score_single_example(self, prompt_template, example, use_semantic=False):
        """
        Score a single example
        
        Args:
            prompt_template: Template with {question} placeholder
            example: Dict with 'input' and 'target' keys
            use_semantic: Whether to use semantic similarity for scoring
        
        Returns:
            Dict with scoring results
        """
        # Format the prompt
        question_prompt = prompt_template.format(question=example['input'])
        full_prompt = self.format_prompt_with_answer_prefix(question_prompt)
        
        # Call LLM
        llm_output = self.call_llm(full_prompt)
        
        # Parse answer
        parsed_answer = self.parse_answer(llm_output)
        
        # Check correctness
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
                    verbose=False, delay=1.0):
        """
        Score a prompt template on multiple examples
        Longer delay (1 second) for China network stability
        
        Args:
            prompt_template: Template with {question} placeholder
            examples: List of examples with 'input' and 'target'
            use_semantic: Use semantic similarity checking
            verbose: Print detailed results
            delay: Delay between API calls (1 second for China)
        
        Returns:
            Dict with accuracy and detailed results
        """
        results = []
        correct_count = 0
        
        print(f"Processing {len(examples)} examples...")
        
        for i, example in enumerate(examples):
            print(f"Example {i+1}/{len(examples)}...", end=' ')
            
            # Score single example
            result = self.score_single_example(
                prompt_template, 
                example, 
                use_semantic=use_semantic
            )
            
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
                print("✓")
            else:
                print("✗")
            
            # Verbose output
            if verbose and i < 3:  # Print first 3 examples
                print(f"\n--- Example {i+1} Details ---")
                # print(f"Question: {result['question'][:80]}...")
                # print(f"LLM Output: {result['llm_output'][:100]}...")
                print(f"Question: {result['question']}")
                print(f"LLM Output: {result['llm_output']}")
                print(f"Parsed: {result['parsed_answer']}")
                print(f"Truth: {result['ground_truth']}")
                print(f"Correct: {result['is_correct']}\n")
            
            # Rate limiting with longer delay for China
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
    print("Testing BBH Scorer...\n")
    
    # Initialize scorer
    scorer = BBHScorer()
    
    # Test example
    test_example = {
        'input': 'Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?',
        'target': '12/14/1937'
    }
    
    # Test with simple prompt
    simple_prompt = "Q: {question}\nA:"
    
    print("=" * 60)
    print("Test 1: Simple Prompt")
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

    
    def parse_answer(self, llm_output):
        """
        Parse the LLM output to extract the final answer
        Looks for the "####" prefix
        
        Args:
            llm_output: Raw LLM response
        
        Returns:
            Parsed answer string, or None if parsing fails
        """
        # Look for #### prefix
        if "####" in llm_output:
            # Extract everything after ####
            parts = llm_output.split("####")
            if len(parts) > 1:
                answer = parts[-1].strip()
                return answer
        
        # Fallback: if no #### found, try to get last line
        lines = llm_output.strip().split('\n')
        if lines:
            # Check if last line looks like an answer
            last_line = lines[-1].strip()
            # Remove common prefixes
            for prefix in ["Answer:", "A:", "Final answer:", "The answer is"]:
                if last_line.startswith(prefix):
                    last_line = last_line[len(prefix):].strip()
            return last_line
        
        return llm_output.strip()
    
    def check_exact_match(self, parsed_answer, ground_truth):
        """
        Check if parsed answer exactly matches ground truth
        
        Args:
            parsed_answer: Answer extracted from LLM output
            ground_truth: Correct answer
        
        Returns:
            Boolean indicating if answers match
        """
        if not parsed_answer or not ground_truth:
            return False
        
        # Normalize both answers
        parsed_norm = parsed_answer.lower().strip()
        truth_norm = ground_truth.lower().strip()
        
        # Direct match
        if parsed_norm == truth_norm:
            return True
        
        # Check if ground truth is contained in parsed answer
        if truth_norm in parsed_norm:
            return True
        
        # For multiple choice answers like "(A)", "(B)", etc.
        # Extract just the letter
        parsed_letter = self._extract_multiple_choice(parsed_norm)
        truth_letter = self._extract_multiple_choice(truth_norm)
        
        if parsed_letter and truth_letter:
            return parsed_letter == truth_letter
        
        return False
    
    def _extract_multiple_choice(self, text):
        """Extract multiple choice letter from text like '(A)' or 'A'"""
        # Match patterns like (A), (B), or just A, B
        match = re.search(r'\(?([A-E])\)?', text.upper())
        if match:
            return match.group(1)
        return None
    
    def check_semantic_similarity(self, parsed_answer, ground_truth):
        """
        Use a scoring LLM to check semantic similarity
        for free-form text answers
        
        Args:
            parsed_answer: Answer from model
            ground_truth: Correct answer
        
        Returns:
            Boolean indicating semantic similarity
        """
        # First try exact match
        if self.check_exact_match(parsed_answer, ground_truth):
            return True
        
        # Use LLM to judge semantic similarity
        scoring_prompt = f"""Compare these two answers and determine if they are semantically equivalent or convey the same meaning.

Answer 1: {parsed_answer}
Answer 2: {ground_truth}

Are these answers semantically similar? Consider them similar if they express the same core idea, even if worded differently.

Respond with ONLY "YES" or "NO".

####"""
        
        try:
            response = self.scoring_llm.generate_content(
                scoring_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=10,
                ),
                request_options={'timeout': 120}
            )
            
            result = response.text.strip().upper()
            return "YES" in result
            
        except Exception as e:
            print(f"Error in semantic similarity check: {e}")
            # Fallback to exact match
            return False
    
    def score_single_example(self, prompt_template, example, use_semantic=False):
        """
        Score a single example
        
        Args:
            prompt_template: Template with {question} placeholder
            example: Dict with 'input' and 'target' keys
            use_semantic: Whether to use semantic similarity for scoring
        
        Returns:
            Dict with scoring results
        """
        # Format the prompt
        question_prompt = prompt_template.format(question=example['input'])
        full_prompt = self.format_prompt_with_answer_prefix(question_prompt)
        
        # Call LLM
        llm_output = self.call_llm(full_prompt)
        
        # Parse answer
        parsed_answer = self.parse_answer(llm_output)
        
        # Check correctness
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
        Longer delay for network stability
        
        Args:
            prompt_template: Template with {question} placeholder
            examples: List of examples with 'input' and 'target'
            use_semantic: Use semantic similarity checking
            verbose: Print detailed results
            delay: Delay between API calls (longer for China)
        
        Returns:
            Dict with accuracy and detailed results
        """
        results = []
        correct_count = 0
        
        for i, example in enumerate(examples):
            # Score single example
            result = self.score_single_example(
                prompt_template, 
                example, 
                use_semantic=use_semantic
            )
            
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            # Verbose output
            if verbose and i < 3:  # Print first 3 examples
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {result['question'][:80]}...")
                print(f"LLM Output: {result['llm_output'][:100]}...")
                print(f"Parsed: {result['parsed_answer']}")
                print(f"Truth: {result['ground_truth']}")
                print(f"Correct: {result['is_correct']}")
            
            # Rate limiting with longer delay for China
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
    print("Testing BBH Scorer...\n")
    
    # Initialize scorer
    scorer = BBHScorer()
    
    # Test example
    test_example = {
        'input': 'Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?',
        'target': '12/14/1937'
    }
    
    # Test with simple prompt
    simple_prompt = "Q: {question}\nA:"
    
    print("=" * 60)
    print("Test 1: Simple Prompt")
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

        
    def format_prompt_with_answer_prefix(self, prompt_text):
        """
        Add answer formatting instruction to prompt
        This ensures consistent parsing of LLM output
        """
        formatted_prompt = f"""{prompt_text}

IMPORTANT: Output your final answer at the end with the prefix "####". 
For example: #### ANSWER

Your response:"""
        return formatted_prompt
    
    def call_llm(self, prompt, max_tokens=200, temperature=0):
        """
        Call LLM with the prompt
        
        Args:
            prompt: The formatted prompt
            max_tokens: Maximum output tokens
            temperature: Sampling temperature (0 for deterministic)
        
        Returns:
            LLM response text
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""
    
    def parse_answer(self, llm_output):
        """
        Parse the LLM output to extract the final answer
        Looks for the "####" prefix
        
        Args:
            llm_output: Raw LLM response
        
        Returns:
            Parsed answer string, or None if parsing fails
        """
        # Look for #### prefix
        if "####" in llm_output:
            # Extract everything after ####
            parts = llm_output.split("####")
            if len(parts) > 1:
                answer = parts[-1].strip()
                return answer
        
        # Fallback: if no #### found, try to get last line
        lines = llm_output.strip().split('\n')
        if lines:
            # Check if last line looks like an answer
            last_line = lines[-1].strip()
            # Remove common prefixes
            for prefix in ["Answer:", "A:", "Final answer:", "The answer is"]:
                if last_line.startswith(prefix):
                    last_line = last_line[len(prefix):].strip()
            return last_line
        
        return llm_output.strip()
    
    def check_exact_match(self, parsed_answer, ground_truth):
        """
        Check if parsed answer exactly matches ground truth
        
        Args:
            parsed_answer: Answer extracted from LLM output
            ground_truth: Correct answer
        
        Returns:
            Boolean indicating if answers match
        """
        if not parsed_answer or not ground_truth:
            return False
        
        # Normalize both answers
        parsed_norm = parsed_answer.lower().strip()
        truth_norm = ground_truth.lower().strip()
        
        # Direct match
        if parsed_norm == truth_norm:
            return True
        
        # Check if ground truth is contained in parsed answer
        if truth_norm in parsed_norm:
            return True
        
        # For multiple choice answers like "(A)", "(B)", etc.
        # Extract just the letter
        parsed_letter = self._extract_multiple_choice(parsed_norm)
        truth_letter = self._extract_multiple_choice(truth_norm)
        
        if parsed_letter and truth_letter:
            return parsed_letter == truth_letter
        
        return False
    
    def _extract_multiple_choice(self, text):
        """Extract multiple choice letter from text like '(A)' or 'A'"""
        # Match patterns like (A), (B), or just A, B
        match = re.search(r'\(?([A-E])\)?', text.upper())
        if match:
            return match.group(1)
        return None
    
    def check_semantic_similarity(self, parsed_answer, ground_truth):
        """
        Use a scoring LLM to check semantic similarity
        for free-form text answers
        
        Args:
            parsed_answer: Answer from model
            ground_truth: Correct answer
        
        Returns:
            Boolean indicating semantic similarity
        """
        # First try exact match
        if self.check_exact_match(parsed_answer, ground_truth):
            return True
        
        # Use LLM to judge semantic similarity
        scoring_prompt = f"""Compare these two answers and determine if they are semantically equivalent or convey the same meaning.

Answer 1: {parsed_answer}
Answer 2: {ground_truth}

Are these answers semantically similar? Consider them similar if they express the same core idea, even if worded differently.

Respond with ONLY "YES" or "NO".

####"""
        
        try:
            response = self.scoring_llm.generate_content(
                scoring_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=10,
                )
            )
            
            result = response.text.strip().upper()
            return "YES" in result
            
        except Exception as e:
            print(f"Error in semantic similarity check: {e}")
            # Fallback to exact match
            return False
    
    def score_single_example(self, prompt_template, example, use_semantic=False):
        """
        Score a single example
        
        Args:
            prompt_template: Template with {question} placeholder
            example: Dict with 'input' and 'target' keys
            use_semantic: Whether to use semantic similarity for scoring
        
        Returns:
            Dict with scoring results
        """
        # Format the prompt
        question_prompt = prompt_template.format(question=example['input'])
        full_prompt = self.format_prompt_with_answer_prefix(question_prompt)
        
        # Call LLM
        llm_output = self.call_llm(full_prompt)
        
        # Parse answer
        parsed_answer = self.parse_answer(llm_output)
        
        # Check correctness
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
                    verbose=False, delay=0.1):
        """
        Score a prompt template on multiple examples
        
        Args:
            prompt_template: Template with {question} placeholder
            examples: List of examples with 'input' and 'target'
            use_semantic: Use semantic similarity checking
            verbose: Print detailed results
            delay: Delay between API calls (rate limiting)
        
        Returns:
            Dict with accuracy and detailed results
        """
        results = []
        correct_count = 0
        
        for i, example in enumerate(examples):
            # Score single example
            result = self.score_single_example(
                prompt_template, 
                example, 
                use_semantic=use_semantic
            )
            
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            # Verbose output
            if verbose and i < 3:  # Print first 3 examples
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {result['question'][:80]}...")
                print(f"LLM Output: {result['llm_output'][:100]}...")
                print(f"Parsed: {result['parsed_answer']}")
                print(f"Truth: {result['ground_truth']}")
                print(f"Correct: {result['is_correct']}")
            
            # Rate limiting
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
    print("Testing BBH Scorer...\n")
    
    # Initialize scorer
    scorer = BBHScorer()
    
    # Test example
    test_example = {
        'input': 'Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?',
        'target': '12/14/1937'
    }
    
    # Test with simple prompt
    simple_prompt = "Q: {question}\nA:"
    
    print("=" * 60)
    print("Test 1: Simple Prompt")
    print("=" * 60)
    
    result = scorer.score_single_example(simple_prompt, test_example)
    
    print(f"Question: {result['question']}")
    print(f"\nLLM Output:\n{result['llm_output']}")
    print(f"\nParsed Answer: {result['parsed_answer']}")
    print(f"Ground Truth: {result['ground_truth']}")
    print(f"Correct: {result['is_correct']}")
    
    print("\n" + "=" * 60)
    print("Test 2: Improved Prompt")
    print("=" * 60)
    
    improved_prompt = """You are an expert at date calculations.

Q: {question}

Provide the answer in MM/DD/YYYY format.
A:"""
    
    result2 = scorer.score_single_example(improved_prompt, test_example)
    
    print(f"LLM Output:\n{result2['llm_output']}")
    print(f"\nParsed Answer: {result2['parsed_answer']}")
    print(f"Correct: {result2['is_correct']}")
    
    print("\n✅ Scorer test complete!")


if __name__ == "__main__":
    test_scorer()