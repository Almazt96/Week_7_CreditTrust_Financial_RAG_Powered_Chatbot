import os
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

# Create the missing subfolder
pooling_dir = "./local_model/1_Pooling"
os.makedirs(pooling_dir, exist_ok=True)

url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json"
dest = os.path.join(pooling_dir, "config.json")

print("Downloading missing pooling configuration...")
try:
    urllib.request.urlretrieve(url, dest)
    print("Success! 1_Pooling/config.json has been created.")
except Exception as e:
    print(f"Failed: {e}")