"""
NOLA News RSS Crawler - Updated for boxsdk-gen with Box Mount
Scrapes New Orleans civic news and saves to mounted Box folder
"""
import os
import time
import json
import hashlib
import datetime as dt
from pathlib import Path
from urllib.parse import urlparse
import re
import ssl

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import trafilatura

# Fix SSL certificate issues on macOS
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables from .env
load_dotenv()

# Box mount path - this should point to your mounted Box folder
# For example: "/Users/username/Box" or "C:/Users/username/Box"
BOX_PATH = os.environ.get("BOX_PATH", "")
ARTICLES_FOLDER = os.environ.get("BOX_ARTICLES_FOLDER", "news")  # Subfolder within Box

# RSS Feeds - New Orleans civic news sources
FEEDS = [
    "https://veritenews.org/feed/",
    "https://thelensnola.org/feed/",
    "https://neworleanscitybusiness.com/feed/"
]

# Keywords for relevance filtering
KEYWORDS = {
    "city council", "ordinance", "zoning", "budget", "millage", "public works",
    "sewerage & water board", "swbno", "dpw", "planning commission", "permit",
    "tax", "mayor", "poll", "school board", "reform", "bond", "levy",
    "land use", "infrastructure", "litigation", "city hall", "municipal",
    "nopd", "nofd", "crime", "public safety", "drainage", "flooding",
    "affordable housing", "rta", "streetcar", "neighborhood", "city attorney",
    "audit", "tourism", "economic development", "property tax", "blight",
    "sanitation", "street repair", "pothole", "traffic", "parking"
}

def sha16(s: str) -> str:
    """Generate 16-character hash for URL identification"""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def get_box_folder_path() -> Path:
    """Get the full path to the Box working folder"""
    if not BOX_PATH:
        raise ValueError(
            "BOX_PATH not set in .env file. "
            "Please set it to your Box mount location (e.g., /Users/username/Box)"
        )
    
    box_path = Path(BOX_PATH)
    if not box_path.exists():
        raise ValueError(
            f"Box mount path does not exist: {box_path}\n"
            f"Please ensure Box Drive is installed and mounted."
        )
    
    # Create subfolder if specified
    if ARTICLES_FOLDER:
        folder_path = box_path / ARTICLES_FOLDER
        folder_path.mkdir(exist_ok=True)
        return folder_path
    
    return box_path

def load_articles_from_box(box_folder: Path) -> list[dict]:
    """Load existing articles from Box articles.json file"""
    print(f"Loading existing articles from {box_folder}...")
    
    articles_file = box_folder / "articles.json"
    
    if articles_file.exists():
        try:
            print(f"   Found articles.json")
            content = articles_file.read_text(encoding='utf-8')
            print(f"   Read {len(content)} characters")
            
            data = json.loads(content)
            
            if not isinstance(data, list):
                print(f"   Warning: articles.json is not a list, starting fresh")
                data = []
            
            article_count = len(data)
            print(f"   Found {article_count} existing articles")
            
            # Show some IDs for debugging
            if data:
                sample_ids = [a.get('id', 'NO_ID')[:8] for a in data[:3]]
                print(f"   Sample IDs: {sample_ids}")
            
            return data
            
        except json.JSONDecodeError as e:
            print(f"   JSON parsing error: {e}")
            print(f"   Content preview: {content[:200]}")
            raise
    else:
        print("   No existing articles.json found, creating new file...")
        # Create empty articles.json
        articles_file.write_text(json.dumps([], indent=2), encoding='utf-8')
        print(f"   Created new articles.json")
        return []

def get_seen_urls(articles: list[dict]) -> set[str]:
    """Extract seen URL IDs from articles list"""
    return {article.get('id') for article in articles if article.get('id')}

def save_articles_to_box(box_folder: Path, articles: list[dict]):
    """Save complete articles list to Box, overwriting existing file"""
    print(f"\n Saving {len(articles)} total articles to Box...")
    
    if articles:
        sample_ids = [a.get('id', 'NO_ID')[:8] for a in articles[:5]]
        print(f"   First 5 article IDs: {sample_ids}")
    
    articles_file = box_folder / "articles.json"
    
    # Create JSON data
    data = json.dumps(articles, indent=2, ensure_ascii=False)
    print(f"   JSON size: {len(data)} bytes ({len(data)/1024:.1f} KB)")
    
    # Write to file
    articles_file.write_text(data, encoding='utf-8')
    print(f"   Saved articles.json successfully")

def save_markdown_file(box_folder: Path, filename: str, content: bytes):
    """Save a markdown file to Box folder"""
    file_path = box_folder / filename
    file_path.write_bytes(content)
    print(f"   Saved: {filename}")

def fetch_rss_entries():
    """Fetch entries from all RSS feeds"""
    print(f"\n Fetching RSS feeds from {len(FEEDS)} sources...")
    total_entries = 0
    
    for url in FEEDS:
        print(f"   - Fetching: {url}")
        try:
            d = feedparser.parse(url)
            source = d.feed.get("title", urlparse(url).netloc)
            
            # Debug: Check feed status
            if hasattr(d, 'status'):
                print(f"     Status: {d.status}")
            if hasattr(d, 'bozo') and d.bozo:
                print(f"     Feed parsing warning: {d.get('bozo_exception', 'Unknown error')}")
            
            entry_count = len(d.entries)
            total_entries += entry_count
            print(f"     Found {entry_count} entries from {source}")
            
            for e in d.entries:
                yield {
                    "title": (e.get("title") or "").strip(),
                    "url": e.get("link"),
                    "published": e.get("published", ""),
                    "source": source,
                }
        except Exception as e:
            print(f"     Error fetching feed: {e}")
            continue
    
    print(f"\n Total entries fetched: {total_entries}")

def extract_text(url: str) -> tuple[str, str]:
    """Extract article text - tries direct fetch first, falls back to browser"""
    print(f"   Extracting content from: {url}")
    
    # Try trafilatura first (faster, no ChromeDriver issues)
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded, 
                include_comments=False, 
                include_tables=False,
                no_fallback=False
            )
            metadata = trafilatura.extract_metadata(downloaded)
            title = metadata.title if metadata and metadata.title else ""
            
            # If we got good content, return it
            if text and len(text) > 200:
                print(f"   Extracted {len(text)} characters (direct fetch)")
                return title, text
    except Exception as e:
        print(f"   Direct fetch failed: {e}")
    
    # Fallback to Selenium if trafilatura didn't work well
    print(f"   Trying with browser...")
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--window-size=1366,768")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = None
    try:
        # Try to use ChromeDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        print(f"   Browser setup failed: {e}")
        return "", ""
    
    try:
        driver.get(url)
        
        # Wait for article content to load
        try:
            WebDriverWait(driver, 15).until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "article")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[class*='entry-content']"))
                )
            )
            time.sleep(2)  # Give page time to fully render
        except Exception:
            pass  # Fall back to trafilatura on whole page
        
        # Get the page HTML
        html = driver.page_source
        
        # Use trafilatura with more permissive settings
        downloaded = trafilatura.extract(
            html, 
            include_comments=False, 
            include_tables=False,
            include_links=False,
            no_fallback=False
        )
        
        # Extract title and metadata
        metadata = trafilatura.extract_metadata(html)
        title = ""
        
        if metadata:
            title = metadata.title or ""
        
        # If trafilatura failed, try BeautifulSoup fallback
        if not downloaded or len(downloaded) < 100:
            print(f"   Trafilatura extraction weak ({len(downloaded or '')} chars), trying BeautifulSoup fallback...")
            soup = BeautifulSoup(html, "lxml")
            
            # Try to find article content
            article = soup.find("article") or soup.find(class_=re.compile("entry-content|post-content|article-content"))
            
            if article:
                # Remove unwanted elements
                for element in article.find_all(['script', 'style', 'nav', 'aside', 'header', 'footer']):
                    element.decompose()
                
                # Get text
                paragraphs = [p.get_text(strip=True) for p in article.find_all(['p', 'h2', 'h3'])]
                downloaded = "\n\n".join([p for p in paragraphs if len(p) > 20])
            
            # If still nothing, get all paragraphs from page
            if not downloaded or len(downloaded) < 100:
                paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
                downloaded = "\n\n".join([p for p in paragraphs if len(p) > 20])
            
            if not title:
                title_tag = soup.find('title')
                title = title_tag.get_text(strip=True) if title_tag else ""
        
        text = downloaded or ""
        
        print(f"   Extracted {len(text)} characters, title: '{title[:50]}...'")
        
    finally:
        if driver:
            driver.quit()
    
    return title, text

def looks_relevant(title: str, text: str) -> bool:
    """Check if article is relevant based on keywords"""
    blob = f"{title}\n{text}".lower()
    matched_keywords = [k for k in KEYWORDS if k in blob]
    
    if matched_keywords:
        print(f"   Relevant (matched: {', '.join(matched_keywords[:3])}{'...' if len(matched_keywords) > 3 else ''})")
        return True
    else:
        print(f"   Not relevant (no keyword matches)")
        return False

def record_md(rec: dict, full_text: str) -> bytes:
    """Generate markdown file content"""
    md = f"""---
source: {rec['source']}
title: {rec['title']}
url: {rec['url']}
published: {rec.get('published', '')}
saved_at: {rec['saved_at']}
---

{full_text}
"""
    return md.encode("utf-8")

def main():
    """Main crawler function"""
    print("\n" + "="*70)
    print("  Crawler Crawling...")
    print("="*70)
    
    # Get Box folder path
    try:
        box_folder = get_box_folder_path()
        print(f" Box folder: {box_folder}")
    except ValueError as e:
        print(str(e))
        print("\n Setup Instructions:")
        print("   1. Install Box Drive: https://www.box.com/resources/downloads")
        print("   2. Sign in and ensure Box is mounted on your system")
        print("   3. Add to your .env file:")
        print("      BOX_PATH=/Users/yourname/Box  # macOS/Linux")
        print("      BOX_PATH=C:/Users/yourname/Box  # Windows")
        print("      ARTICLES_FOLDER=NOLA_News  # Optional subfolder")
        return
    
    # Load existing articles from Box (this is our source of truth)
    articles = load_articles_from_box(box_folder)
    seen = get_seen_urls(articles)
    print(f" Currently tracking {len(seen)} unique articles")
    
    new_articles = []
    processed_count = 0
    skipped_seen = 0
    skipped_no_url = 0
    
    # Process RSS entries
    for item in fetch_rss_entries():
        url = item["url"]
        if not url:
            skipped_no_url += 1
            print(f"Skipping entry with no URL: {item['title'][:50]}")
            continue
        
        key = sha16(url)
        if key in seen:
            skipped_seen += 1
            continue
        
        processed_count += 1
        print(f"\n{'='*70}")
        print(f"[{processed_count}] Processing: {item['title'][:60]}...")
        print(f"   Source: {item['source']}")
        print(f"   URL: {url}")
        
        try:
            # Extract article content
            title_ext, text = extract_text(url)
            title = title_ext or item["title"]
            
            if not text or len(text) < 100:
                print(f"   Skipping - insufficient content ({len(text)} chars)")
                continue
            
            # Check relevance
            if not looks_relevant(title, text):
                continue
            
            full_text = (text or "").strip()
            
            # Create article record
            rec = {
                "id": key,
                "source": item["source"],
                "url": url,
                "title": title,
                "published": item.get("published", ""),
                "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "content_preview": full_text[:200] + "..." if len(full_text) > 200 else full_text
            }
            
            # Generate filename
            published_date = rec["published"][:10] if rec.get("published") else dt.date.today().isoformat()
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_title = clean_title[:50]
            md_filename = f"{published_date}_{key}_{clean_title}.md"
            
            # Save markdown file to Box
            try:
                save_markdown_file(box_folder, md_filename, record_md(rec, full_text))
                print(f"   Saved to Box")
            except Exception as save_err:
                print(f"   Box save failed: {save_err}")
                continue
            
            # Add to tracking
            new_articles.append(rec)
            seen.add(key)
            
            # IMMEDIATELY save to Box after each successful article
            # This prevents data loss if user interrupts
            articles.append(rec)
            try:
                save_articles_to_box(box_folder, articles)
            except Exception as e:
                print(f"   Failed to save state: {e}")
            
            print(f"   Article saved successfully!")
            print(f"   Record ID: {key}")
            
            # Be polite to servers
            print(f"   Waiting 10 seconds...")
            time.sleep(10)
            
        except Exception as e:
            print(f"   Error processing article: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    if new_articles:
        print("\n" + "="*70)
        print(f" Success! Added {len(new_articles)} new articles")
        print("="*70)
        print(f" Total articles tracked: {len(articles)}")
        print(f" Box folder: {box_folder}")
        print(f" State file: articles.json (tracks {len(articles)} articles)")
    else:
        print("\n" + "="*70)
        print("  No new relevant articles found")
        print("="*70)
        print(f" Stats:")
        print(f"   - Total entries fetched: {processed_count + skipped_seen + skipped_no_url}")
        print(f"   - Already seen (skipped): {skipped_seen}")
        print(f"   - No URL (skipped): {skipped_no_url}")
        print(f"   - New entries processed: {processed_count}")
        print(f"   - Passed relevance filter: {len(new_articles)}")
        print(f"State: {len(articles)} articles tracked in articles.json")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Crawler interrupted by user")
    except Exception as e:
        print(f"\n\n Fatal error: {e}")
        raise