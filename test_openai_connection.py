import os
import requests
import json

# Configuration from .opencode/config.json
API_KEY = "sk_MWhPzZXMBVJeKcnAO9hL-cxQDmM7t2aMknf8tC_6tbQ"
BASE_URL = "https://api.jiekou.ai/openai/v1"

def test_model(model_name):
    print(f"\n--- Testing Model: {model_name} ---")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Simple prompt
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_completion_tokens": 1000
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            print("✅ Success!")
            print(f"Response: {response.json()['choices'][0]['message']['content']}")
            return True
        else:
            print(f"❌ Failed (Status {response.status_code})")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    # 1. Test o3 (Primary Target - Verified Available)
    success = test_model("o3")
    
    # 2. If o3 fails, try o3-mini (Alternative)
    if not success:
        test_model("o3-mini")
        
    # 3. If that fails, try o1-mini
    if not success:
        success = test_model("o1-mini")
        
    # 4. If that fails, try gpt-4o to verify connectivity
    if not success:
        test_model("gpt-4o")
