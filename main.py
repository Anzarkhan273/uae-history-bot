import os
import pickle
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Loading embedding model...")
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Loading saved knowledge base...")
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    chunk_objects = data["chunk_objects"]
    chunk_embeddings = data["embeddings"]


def find_relevant_chunks(question, top_k=3):
    question_embedding = embed_model.encode([question])[0]
    similarities = np.dot(chunk_embeddings, question_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunk_objects[i] for i in top_indices]


def ask_bot(question):
    relevant = find_relevant_chunks(question)
    context = "\n\n".join([c["text"] for c in relevant])
    sources = list(set([c["source"] for c in relevant]))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable guide on UAE history. "
                    "Use the provided context to answer accurately. "
                    "If the context doesn't fully answer the question, say so honestly rather than guessing.\n\n"
                    "Respond in two parts:\n"
                    "1. Answer in Arabic\n"
                    "2. Then provide the English translation of that same answer\n\n"
                    f"Context:\n{context}"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    answer = response.choices[0].message.content
    return answer, sources


if __name__ == "__main__":
    print("\nUAE History Bot (RAG-powered) — ask me anything (type 'exit' to quit)\n")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer, sources = ask_bot(question)
        print(f"\nBot:\n{answer}")
        print(f"\n📚 Sources: {', '.join(sources)}\n")