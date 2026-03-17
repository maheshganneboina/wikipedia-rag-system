import bz2
import xml.etree.ElementTree as ET
import json
import re

input_file = "enwiki-20260301-pages-articles-multistream.xml.bz2"
output_file = "wiki_articles.json"

articles = []
max_articles = 100


def clean_wiki_text(text: str) -> str:
    if not text:
        return ""

    # Skip redirect pages
    if text.strip().upper().startswith("#REDIRECT"):
        return ""

    # Remove ref tags
    text = re.sub(r"<ref[^>/]*>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<ref[^>]*/>", "", text, flags=re.IGNORECASE)

    # Remove templates {{...}}
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)

    # Convert wiki links [[Page|Text]] -> Text
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)

    # Remove external links [http://... text] -> text
    text = re.sub(r"\[https?://[^\s\]]+\s([^\]]+)\]", r"\1", text)

    # Remove bold/italic markup
    text = text.replace("'''", "").replace("''", "")

    # Remove headings markup == Heading ==
    text = re.sub(r"=+\s*(.*?)\s*=+", r"\1", text)

    # Remove HTML/XML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove leftover table/category/file markup lines
    text = re.sub(r"\[\[(Category|File|Image):.*?\]\]", "", text, flags=re.IGNORECASE)

    # Collapse whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


with bz2.open(input_file, "rb") as f:
    context = ET.iterparse(f, events=("end",))

    for event, elem in context:
        if elem.tag.endswith("page"):
            title = None
            text = None

            for child in elem:
                if child.tag.endswith("title"):
                    title = child.text

                if child.tag.endswith("revision"):
                    for rev_child in child:
                        if rev_child.tag.endswith("text"):
                            text = rev_child.text
                            break

            cleaned_text = clean_wiki_text(text)

            if title and cleaned_text and len(cleaned_text) > 300:
                articles.append({
                    "title": title,
                    "text": cleaned_text
                })
                print(f"Extracted: {title}")

            elem.clear()

            if len(articles) >= max_articles:
                break

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(articles)} cleaned articles to {output_file}")
