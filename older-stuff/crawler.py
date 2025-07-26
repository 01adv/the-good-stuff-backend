import os
import re
import json
import time
import logging
import urllib.parse
from typing import Set, Dict, Any, List, Optional

import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

BASE_URL = "https://thegood.com/"
DOMAIN = "thegood.com"
START_URLS = [BASE_URL, "https://thegood.com/insights/", "https://thegood.com/case-studies/"]
MAX_DEPTH = 2
OUTPUT_FILE = "output.jsonl"
REQUEST_DELAY = 1  # seconds

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- OpenAI Embedding Disabled ---
logging.info("Embeddings and OpenAI integration are disabled. Only website data will be crawled and saved.")

# --- Helper Functions ---

def get_page_html(url: str) -> Optional[str]:
    """Fetches the HTML content of a page with error handling."""
    try:
        time.sleep(REQUEST_DELAY)
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def clean_text(soup) -> str:
    """Removes scripts, styles, nav, footers and collapses whitespace."""
    # Accepts BeautifulSoup or Tag
    if not hasattr(soup, 'find_all'):
        return str(soup).strip()
    for element in soup(["script", "style", "nav", "footer", "header", ".header", ".footer"]):
        element.decompose()
    text = soup.get_text(separator=' ', strip=True)
    text = re.sub(r'\s+', ' ', text) # Collapse whitespace
    return text.strip()

def create_slug(url: str) -> str:
    """Creates a URL-safe slug from a URL."""
    path = urllib.parse.urlparse(url).path
    slug = path.strip('/').replace('/', '_')
    return slug if slug else "homepage"

## Embedding function removed

# --- Main Crawler Logic ---

def crawl_site(start_urls: List[str], max_depth: int):
    """Crawls a website, extracts content, and saves it to a JSONL file."""
    visited_urls: Set[str] = set()
    urls_to_visit: List[Dict[str, Any]] = [{"url": url, "depth": 0} for url in start_urls]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=50,
        length_function=len,
    )

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        while urls_to_visit:
            current = urls_to_visit.pop(0)
            url, depth = current["url"], current["depth"]

            if url in visited_urls or not url.startswith(BASE_URL):
                continue

            logging.info(f"Crawling (depth {depth}): {url}")
            visited_urls.add(url)

            html = get_page_html(url)
            if not html:
                continue

            soup = BeautifulSoup(html, 'html.parser')

            # Robust page title extraction
            page_title = "No Title"
            if soup.title:
                if soup.title.string:
                    page_title = soup.title.string.strip()
                else:
                    page_title = soup.title.get_text(strip=True)

            # Extract headers and main content
            main_content_soup = soup.find('main') or soup.find('article') or soup
            # Ensure main_content_soup is a Tag or BeautifulSoup
            if not hasattr(main_content_soup, 'find_all'):
                main_content_soup = soup


            headers = []
            if hasattr(main_content_soup, 'find_all') and not isinstance(main_content_soup, NavigableString):
                for h in main_content_soup.find_all(['h1', 'h2', 'h3']):
                    try:
                        headers.append(h.get_text(strip=True))
                    except Exception:
                        headers.append(str(h))
            content_text = clean_text(main_content_soup)

            full_text = " ".join(headers) + " " + content_text
            chunks = text_splitter.split_text(full_text)

            url_slug = create_slug(url)

            for i, chunk_text in enumerate(chunks):
                record = {
                    "id": f"{url_slug}_{i}",
                    "url": url,
                    "title": page_title,
                    "chunk": chunk_text,
                }
                f.write(json.dumps(record) + '\n')

            if depth < max_depth:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urllib.parse.urljoin(BASE_URL, href).split('#')[0] # Normalize and remove fragment
                    
                    if full_url not in visited_urls and DOMAIN in urllib.parse.urlparse(full_url).netloc:
                        urls_to_visit.append({"url": full_url, "depth": depth + 1})

    logging.info(f"Crawling complete. Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    crawl_site(START_URLS, MAX_DEPTH)
