# UAE History Bot (Bilingual RAG Chatbot)

A conversational AI agent that answers questions about UAE history in both Arabic and English, grounded in real Wikipedia source content rather than relying purely on model memory.

## What it does

- Ask any question about UAE history in natural language
- Get a response in Arabic, followed by the English translation
- Answers are grounded using Retrieval-Augmented Generation (RAG) — the bot retrieves relevant passages from a curated Wikipedia knowledge base before generating its answer
- Each answer cites which source article it pulled from

## Why RAG, not just a plain LLM call

A plain LLM call relies entirely on the model's training memory, which can be inconsistent or outdated, especially for regional history topics. This project instead:
1. Retrieves the most relevant real source text for the question
2. Feeds that text to the model as context
3. Has the model answer based on that grounded context, and say so honestly if the context doesn't fully cover the question

This makes answers more verifiable and reduces hallucination risk compared to relying on model memory alone.

## Architecture

User question
│
▼
Embed question (multilingual MiniLM, local)
│
▼
Compare to pre-embedded Wikipedia chunks (cosine similarity)
│
▼
Retrieve top 3 most relevant chunks + their sources
│
▼
Feed question + retrieved context to Llama 3.3 70B (via Groq API)
│
▼
Bilingual answer (Arabic + English) + source citation

## Tech stack

- **LLM**: Llama 3.3 70B, served via Groq API (free tier)
- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (Sentence Transformers, runs locally, supports Arabic + English)
- **Knowledge base**: Wikipedia articles on UAE history (English + Arabic), fetched via Wikipedia API
- **Language**: Python

## Project structure
uae-history-bot/
├── fetch_wikipedia.py   # Pulls UAE history articles from Wikipedia (EN + AR)
├── chunk.py             # Splits articles into overlapping text chunks with source tracking
├── embed_chunks.py      # Converts chunks into embeddings for semantic search
├── main.py              # Interactive chatbot — the main entry point
├── uae_history_data.txt # Raw fetched Wikipedia text
├── chunks.json          # Chunked text with source labels
├── embeddings.pkl       # Precomputed embeddings for retrieval
├── requirements.txt     # Python dependencies
└── .env                 # API key (not committed to version control)

## How to run it

1. Clone this repo and create a virtual environment:
python -m venv venv
.\venv\Scripts\Activate.ps1
2. Install dependencies:
pip install -r requirements.txt
3. Get a free API key from [console.groq.com](https://console.groq.com) and add it to a `.env` file:
GROQ_API_KEY=your_key_here
4. (First time only) Build the knowledge base:
python fetch_wikipedia.py
python chunk.py
python embed_chunks.py
5. Run the chatbot:
python main.py

## Known limitations / future improvements

- Knowledge base currently covers a small, curated set of Wikipedia articles — not comprehensive
- No handling for dialect-specific Arabic phrasing (currently assumes Modern Standard Arabic)
- No conversation memory across turns — each question is answered independently
- Could add a confidence/uncertainty flag when retrieved context is weak
- Could expand source coverage (government heritage sites, academic sources) for stronger grounding

## Author

Anzar Khan