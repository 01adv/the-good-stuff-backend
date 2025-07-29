from bs4 import BeautifulSoup
import requests
import json
from itertools import islice
import re
# from nltk.tokenize import sent_tokenize
import time
from typing import List, Dict

# Replace nltk's sent_tokenize with this
def sent_tokenize(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

# Helper Functions (from previous discussions)
def clean_text(text: str) -> str:
    """Clean text by normalizing whitespace and removing HTML tags."""
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'<.*?>', '', text)  # Remove any lingering HTML tags
    return text.strip()

def chunk_text(text: str, max_tokens: int = 250, overlap_ratio: float = 0.15) -> List[str]:
    """
    Chunk text into pieces for vector embedding.
    - max_tokens: Approximate max words per chunk.
    - overlap_ratio: Fraction of previous chunk to overlap (e.g., 0.15 for 15%).
    """
    sentences = sent_tokenize(text)
    chunks = []
    chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            if chunk:
                chunks.append(' '.join(chunk))
            # Add overlap for context
            overlap_count = int(len(chunk) * overlap_ratio)
            chunk = chunk[-overlap_count:] if overlap_count > 0 else []
            current_length = sum(len(s.split()) for s in chunk)

        chunk.append(sentence)
        current_length += sentence_length

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

def extract_full_content(url: str) -> str:
    """Extract full text content from a given URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; DataScraper/1.0)'}  # Polite user-agent
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract all paragraphs and combine text
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs)
        return text if text else ""
    except Exception:
        return ""

def extract_content(url: str, subcategory: str) -> List[Dict]:
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; DataScraper/1.0)'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    data = []

    # Extract Date (if available, e.g., from <time> or meta tags)
    date_elem = soup.find('time') or soup.find('meta', attrs={'name': 'date'})
    date = date_elem['datetime'] if date_elem and 'datetime' in date_elem.attrs else date_elem.get('content') if date_elem else None

    # Extract Introduction (treat as use_case content_type)
    heading = soup.find('h1').get_text(strip=True) if soup.find('h1') else ""
    intro_paragraph = soup.find('p').get_text(strip=True) if soup.find('p') else ""
    intro_text = clean_text(f"{heading}\n{intro_paragraph}")
    intro_chunks = chunk_text(intro_text)
    for idx, chunk in enumerate(intro_chunks):
        data.append({
            "title": heading,
            "category": "use_case",  # Mapped from "introduction" for bucketing
            "use_case": subcategory,
            "date": date,
            "chunk_index": idx,
            "text_chunk": chunk,
            "url": url
        })

    # Extract Case Studies
    case_study_section = soup.find_all(class_='wp-block-stackable-column')
    for case in case_study_section:
        title = case.find(class_='stk-block-heading__text').get_text(strip=True) if case.find(class_='stk-block-heading__text') else ""
        summary = case.find(class_='stk-block-text__text').get_text(strip=True) if case.find(class_='stk-block-text__text') else ""
        link = case.find('a')['href'] if case.find('a') else url
        if title and summary:  # Ensure valid case study
            full_content = extract_full_content(link) if link.startswith("https://thegood.com/results/") else ""
            combined_text = clean_text(f"{title}\n{summary}\n{full_content}")
            chunks = chunk_text(combined_text)
            for idx, chunk in enumerate(chunks):
                data.append({
                    "title": title,
                    "category": "case_study",
                    "use_case": subcategory,
                    "date": date,
                    "chunk_index": idx,
                    "text_chunk": chunk,
                    "url": link
                })

    # Extract Insights
    insight_section = soup.find_all(class_='stk-block-posts__item')
    for insight in insight_section:
        title = insight.find(class_='stk-block-posts__title').find('a').get_text(strip=True) if insight.find(class_='stk-block-posts__title') else ""
        summary = insight.find(class_='stk-block-posts__excerpt').find('p').get_text(strip=True) if insight.find(class_='stk-block-posts__excerpt') else ""
        link = insight.find('a')['href'] if insight.find('a') else url
        if title and summary:  # Ensure valid insight
            full_content = extract_full_content(link) if link.startswith("https://thegood.com/insights/") else ""
            combined_text = clean_text(f"{title}\n{summary}\n{full_content}")
            chunks = chunk_text(combined_text)
            for idx, chunk in enumerate(chunks):
                data.append({
                    "title": title,
                    "category": "insight",
                    "use_case": subcategory,
                    "date": date,
                    "chunk_index": idx,
                    "text_chunk": chunk,
                    "url": link
                })

    time.sleep(1)  # Delay to be polite to the server
    return data

# Crawl all subcategories
subcategories = [
    ("https://thegood.com/increase-registration/", "increase_registration"),
    ("https://thegood.com/improve-onboarding/", "improve_onboarding"),
    ("https://thegood.com/monetize-free-users/", "monetize_free_users"),
    ("https://thegood.com/improve-retention/", "improve_retention"),
    ("https://thegood.com/increase-referrals/", "increase_referrals"),
    ("https://thegood.com/mitigate-cancellations/", "mitigate_cancellations")
]

all_data = []
for url, subcategory in subcategories:
    all_data.extend(extract_content(url, subcategory))

# Save to JSON
with open("structured_data_latest.json", "w") as f:
    json.dump(all_data, f, indent=2)

print(f"Extracted and structured {len(all_data)} chunks.")
