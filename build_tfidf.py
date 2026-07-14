import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

with open("chunks.json", "r", encoding="utf-8") as f:
    chunk_objects = json.load(f)

texts = [c["text"] for c in chunk_objects]

print(f"Building TF-IDF index for {len(texts)} chunks...")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

with open("tfidf_data.pkl", "wb") as f:
    pickle.dump({
        "chunk_objects": chunk_objects,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix
    }, f)

print("Done! Saved to tfidf_data.pkl")