"""
Check which Google Gemini models are available
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

print("Checking available models...\n")

try:
    models = genai.list_models()
    
    print("Available models that support generateContent:\n")
    print("="*60)
    
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            print(f"Name: {model.name}")
            print(f"Display Name: {model.display_name}")
            print(f"Description: {model.description}")
            print("-"*60)
    
except Exception as e:
    print(f"Error: {e}")
    print("\nThis might indicate network issues (Great Firewall blocking)")