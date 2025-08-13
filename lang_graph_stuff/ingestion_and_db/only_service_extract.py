from bs4 import BeautifulSoup
import requests
import re
from typing import List, Dict

def clean_text(text: str) -> str:
    """Normalize whitespace and strip HTML artifacts."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_use_case_sections(url: str, subcategory: str) -> List[Dict]:
    """
    Extracts structured section data from service/use-case landing pages.
    Focuses on extracting headings, descriptive paragraphs, and list items.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; DataScraper/1.0)'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    sections_data = []

    # Look for content-rich sections commonly used in marketing pages
    content_blocks = soup.find_all(['section', 'div'], class_=re.compile(r'(stk-block|has-background|section|row)'))

    for idx, block in enumerate(content_blocks):
        # Extract heading text (if any)
        heading_tag = block.find(['h2', 'h3', 'h4'])
        heading = heading_tag.get_text(strip=True) if heading_tag else ""

        # Extract paragraph and list text
        body_elements = block.find_all(['strong', 'p'])
        body_text = ' '.join([el.get_text(strip=True) for el in body_elements])
        body_text = clean_text(body_text)

        # Skip if content is too thin to be meaningful or if there's no heading
        if len(body_text) < 50 or not heading:
            continue

        # Skip if this heading has already been added
        if any(section["title"] == heading for section in sections_data):
            continue

        full_section_text = clean_text(f"{heading}\n{body_text}")

        sections_data.append({
            "title": heading,
            "category": "use_case",
            "use_case": subcategory,
            "chunk_index": idx,
            "text_chunk": full_section_text,
            "url": url
        })

    return sections_data


service_urls = [
     ("https://thegood.com/increase-registration/", "increase_registration"),
    ("https://thegood.com/improve-onboarding/", "improve_onboarding"),
    ("https://thegood.com/monetize-free-users/", "monetize_free_users"),
    ("https://thegood.com/improve-retention/", "improve_retention"),
    ("https://thegood.com/increase-referrals/", "increase_referrals"),
    ("https://thegood.com/mitigate-cancellations/", "mitigate_cancellations"),
]


all_use_case_data = []
for url, slug in service_urls:
    print(f"Scraping: {slug}")
    all_use_case_data.extend(extract_use_case_sections(url, slug))

# Save to JSON
import json
with open("use_case_sections.json", "w") as f:
    json.dump(all_use_case_data, f, indent=2)

print(f"Extracted {len(all_use_case_data)} use_case chunks.")
