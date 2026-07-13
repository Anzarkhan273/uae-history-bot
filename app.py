import os
import pickle
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="UAE History Bot", page_icon="UAE")

@st.cache_resource
def load_resources():
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    with open("embeddings.pkl", "rb") as f:
        data = pickle.load(f)
    return client, embed_model, data["chunk_objects"], data["embeddings"]

client, embed_model, chunk_objects, chunk_embeddings = load_resources()

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
                    "If the context doesn't fully answer the question, say so honestly rather than guessing.

"
                    "Respond in two parts:
"
                    "1. Answer in Arabic
"
                    "2. Then provide the English translation of that same answer

"
                    "Context:
" + context
                )
            },
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content, sources

st.title("UAE History Bot")
st.caption("Ask anything about UAE history - answers in Arabic and English, grounded in real sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about UAE history...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask_bot(question)
            st.markdown(answer)
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append({"role": "assistant", "content": answer})
