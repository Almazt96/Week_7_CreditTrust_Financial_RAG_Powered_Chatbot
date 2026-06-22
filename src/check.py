import chromadb
from src.config import VECTOR_STORE_DIR, COLLECTION_NAME

client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
# Check if your collection actually exists and has items
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"📦 Collection Found: {COLLECTION_NAME}")
    print(f"🔢 Total Document Chunks Stored: {collection.count()}")
except Exception as e:
    print(f"❌ Error: {e}")