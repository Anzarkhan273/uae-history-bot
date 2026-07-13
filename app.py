import os
import pickle
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="UAE History Bot", page_icon="🇦🇪")


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
    sources = list(set([c["source"] for c in