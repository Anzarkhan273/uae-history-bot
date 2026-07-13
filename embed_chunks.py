import json
from sentence_transformers import SentenceTransformer
import pickle

with open("chunks.json", "r", encoding="utf-8") as f:
    chunk_objects = json.load(f)

# Extract just the text for embedding, keep full objects for later
texts = [c["text"] for c in chunk_objects]

print(f"Loading embedding model... (first run downloads it, may take a minute)")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print(f"Creating embeddings for {len(texts)} chunks...")
embeddings = model.encode(texts, show_progress_bar=True)

with open("embeddings.pkl", "wb") as f:
    pickle.dump({"chunk_objects": chunk_objects, "embeddings": embeddings}, f)

print("Done! Saved to embeddings.pkl")