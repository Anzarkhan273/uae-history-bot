import wikipediaapi

def fetch_article(title, lang):
    wiki = wikipediaapi.Wikipedia(user_agent="uae-history-bot", language=lang)
    page = wiki.page(title)
    if page.exists():
        return page.text
    else:
        print(f"Page not found: {title} ({lang})")
        return ""

# Topics to pull — same topics in both languages
topics_en = [
    "History of the United Arab Emirates",
    "Trucial States",
    "Zayed bin Sultan Al Nahyan",
    "Pearl hunting",
]

topics_ar = [
    "تاريخ الإمارات العربية المتحدة",
    "الإمارات المتصالحة",
    "زايد بن سلطان آل نهيان",
    "الغوص على اللؤلؤ",
]

if __name__ == "__main__":
    all_text = ""

    print("Fetching English articles...")
    for topic in topics_en:
        text = fetch_article(topic, "en")
        all_text += f"\n\n--- {topic} (English) ---\n{text}"

    print("Fetching Arabic articles...")
    for topic in topics_ar:
        text = fetch_article(topic, "ar")
        all_text += f"\n\n--- {topic} (Arabic) ---\n{text}"

    with open("uae_history_data.txt", "w", encoding="utf-8") as f:
        f.write(all_text)

    print("Done! Saved to uae_history_data.txt")