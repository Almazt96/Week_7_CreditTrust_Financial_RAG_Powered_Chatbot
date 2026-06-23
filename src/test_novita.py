import os
import requests

# Grab the key exactly how your app does (adjust if using a custom key variable)
api_key = os.getenv("NOVITA_API_KEY") 

url = "https://api.novita.ai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "meta-llama/llama-3-8b-instruct", # Change to the model you use in rag_pipeline
    "messages": [{"role": "user", "content": "Hello"}]
}

try:
    print("Testing outbound connection to Novita AI...")
    response = requests.post(url, json=data, headers=headers, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Connection failed directly: {e}")