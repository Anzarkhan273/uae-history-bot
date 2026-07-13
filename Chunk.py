import json

def chunk_text_with_source(text, source_name, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source_name})
        start += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    with open("uae_history_data.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    sections = full_text.split("--- ")
    all_chunks = []

    for section in sections:
        if not section.strip():
            continue
        lines = section.split(" ---\n", 1)
        if len(lines) == 2:
            source_name, content = lines
            chunks = chunk_text_with_source(content, source_name.strip())
            all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")
    print("\nExample chunk:\n")
    print(all_chunks[5] if len(all_chunks) > 5 else all_chunks[0])

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("\nSaved all chunks (with sources) to chunks.json")