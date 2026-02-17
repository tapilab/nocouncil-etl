"""
Articles Vectorizer
Adds news articles collection to existing ChromaDB
Reads from Box-mounted folders, writes to local ChromaDB
"""
import os
import sys
import warnings

# Disable ChromaDB telemetry BEFORE importing chromadb
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Suppress ChromaDB telemetry warnings
warnings.filterwarnings('ignore', message='.*telemetry.*')

import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function \
    import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def load_articles_from_json(json_path: Path) -> List[Dict]:
    """
    Load articles from articles.json file.
    
    Args:
        json_path: Path to articles.json
        
    Returns:
        List of article dictionaries
    """
    if not json_path.exists():
        print(f"   File not found: {json_path}")
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        if not isinstance(articles, list):
            print(f"   JSON file is not a list")
            return []
        
        print(f"   Loaded {len(articles)} articles from JSON")
        return articles
    
    except Exception as e:
        print(f"   Error loading JSON: {e}")
        return []


def parse_markdown_frontmatter(md_path: Path) -> tuple[Dict, str]:
    """
    Parse markdown file with YAML frontmatter.
    
    Returns:
        (metadata_dict, content_text)
    """
    content = md_path.read_text(encoding='utf-8')
    
    # Check for frontmatter
    if not content.startswith('---'):
        return {}, content
    
    # Split frontmatter and content
    parts = content.split('---', 2)
    if len(parts) < 3:
        return {}, content
    
    frontmatter = parts[1].strip()
    text = parts[2].strip()
    
    # Parse frontmatter (simple key: value parser)
    metadata = {}
    for line in frontmatter.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
    
    return metadata, text


def load_articles_from_markdown(folder_path: Path) -> List[Dict]:
    """
    Load articles from individual markdown files.
    Parses frontmatter and content.
    
    Args:
        folder_path: Path to folder containing .md files
        
    Returns:
        List of article dictionaries with 'content' field
    """
    if not folder_path.exists():
        print(f"   Folder not found: {folder_path}")
        return []
    
    md_files = list(folder_path.glob("*.md"))
    
    if not md_files:
        print(f"   No .md files found in {folder_path}")
        return []
    
    articles = []
    
    for md_path in md_files:
        try:
            metadata, content = parse_markdown_frontmatter(md_path)
            
            if not content.strip():
                continue
            
            # Build article dict
            article = {
                'content': content,
                'title': metadata.get('title', md_path.stem),
                'url': metadata.get('url', ''),
                'source': metadata.get('source', ''),
                'published': metadata.get('published', ''),
                'saved_at': metadata.get('saved_at', ''),
                'filename': md_path.name,
            }
            
            articles.append(article)
            
        except Exception as e:
            print(f"   Error parsing {md_path.name}: {e}")
            continue
    
    print(f"   Loaded {len(articles)} articles from markdown files")
    return articles


def vectorize_articles(collection, box_path: str, articles_folder: str, use_markdown: bool = True):
    """
    Vectorize news articles from articles.json and/or markdown files.
    
    Args:
        collection: ChromaDB collection for articles
        box_path: Path to Box root folder
        articles_folder: Relative folder name or full path to articles
        use_markdown: If True, also load from individual .md files
    """
    print("\n" + "="*70)
    print("📰 Vectorizing News Articles")
    print("="*70)
    
    # Handle articles folder path
    if '/' in articles_folder or '\\' in articles_folder:
        # Full path provided
        folder_path = Path(articles_folder)
    else:
        # Relative folder name - combine with box_path
        folder_path = Path(box_path) / articles_folder
    
    print(f"📁 Articles folder: {folder_path}")
    
    if not folder_path.exists():
        print(f" Articles folder not found: {folder_path}")
        print("   Make sure you've run the RSS crawler first")
        return collection
    
    articles = []
    
    # Load from articles.json
    json_path = folder_path / 'articles.json'
    if json_path.exists():
        print(f"📥 Loading from {json_path.name}")
        articles_from_json = load_articles_from_json(json_path)
        articles.extend(articles_from_json)
    else:
        print(f" No articles.json found at {json_path}")
    
    # Load from markdown files
    if use_markdown:
        print(f" Loading markdown files from {folder_path.name}/")
        articles_from_md = load_articles_from_markdown(folder_path)
        articles.extend(articles_from_md)
    
    if not articles:
        print(" No articles found to vectorize")
        return collection
    
    # Filter out articles without content
    articles = [a for a in articles if a.get('content', '').strip()]
    print(f"Total articles to process: {len(articles)}")
    
    if len(articles) == 0:
        print(' No articles with content found')
        return collection
    
    # Prepare documents and metadata
    print(' Preparing articles for embedding...')
    documents = []
    metadatas = []
    ids = []
    
    for i, article in enumerate(tqdm(articles, desc="Processing articles")):
        content = article.get('content', '')
        
        if not content.strip():
            continue
        
        documents.append(content)
        
        # Build metadata (exclude 'content' field as it's in documents)
        metadata = {
            'title': article.get('title', ''),
            'url': article.get('url', ''),
            'source': article.get('source', ''),
            'published': article.get('published', ''),
            'saved_at': article.get('saved_at', ''),
            'filename': article.get('filename', ''),
        }
        
        # Remove empty values
        metadata = {k: v for k, v in metadata.items() if v}
        metadatas.append(metadata)
        
        # Create unique ID
        article_id = article.get('id') or article.get('url') or f"article_{i}"
        ids.append(str(article_id))
    
    # Add to collection in batches
    print('🔄 Adding to ChromaDB...')
    batch_size = 500
    total_added = 0
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        
        total_added += len(batch_docs)
    
    print(f'Done! Added {total_added} articles')
    return collection


def main():
    """
    Main vectorization script for articles only.
    """
    print("\n" + "="*70)
    print(" News Articles Vectorizer")
    print("="*70)
    
    load_dotenv()
    
    # Get configuration
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR")
    BOX_PATH = os.getenv("BOX_PATH")
    BOX_ARTICLES_FOLDER = os.getenv("BOX_ARTICLES_FOLDER", "articles")
    
    # Validate required variables
    if not CHROMA_DB_DIR:
        print("\n CHROMA_DB_DIR not set in .env file")
        print("   This should point to your existing ChromaDB location")
        return
    
    if not BOX_PATH:
        print("\n BOX_PATH not set in .env file")
        print("   This should point to your Box mount location")
        return
    
    print(f"\n📁 Configuration:")
    print(f"   ChromaDB: {CHROMA_DB_DIR}")
    print(f"   Box Path: {BOX_PATH}")
    print(f"   Articles Folder: {BOX_ARTICLES_FOLDER}")
    
    # Validate paths exist
    chroma_path = Path(CHROMA_DB_DIR)
    box_path = Path(BOX_PATH)
    
    print(f"\n🔍 Validating paths...")
    
    if not chroma_path.exists():
        print(f"   ChromaDB directory does not exist: {CHROMA_DB_DIR}")
        print(f"   Creating new database...")
        chroma_path.mkdir(parents=True, exist_ok=True)
    else:
        print(f"   ChromaDB exists")
        
    if not box_path.exists():
        print(f"   BOX_PATH does not exist: {BOX_PATH}")
        print(f"   Make sure Box Drive is mounted")
        return
    else:
        print(f"   BOX_PATH exists")
    
    # Initialize ChromaDB client
    print(f"\n Initializing ChromaDB...")
    chroma_client = PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Check existing collections
    existing_collections = chroma_client.list_collections()
    if existing_collections:
        print(f"   Existing collections: {[c.name for c in existing_collections]}")
        for c in existing_collections:
            print(f"      {c.name}: {c.count()} items")
    else:
        print(f"   No existing collections (this will be the first)")
    
    # Initialize embedding function
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",  # small, fast, 384-dim
        device="cpu",                    # or "cuda" if GPU available
        normalize_embeddings=True
    )
    print(f"   Using embedding model: all-MiniLM-L6-v2")
    
    # Create or get articles collection
    collection_articles = chroma_client.get_or_create_collection(
        name="articles",
        embedding_function=embed_fn,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:num_threads": 1
        }
    )
    
    # Check if collection already has items
    existing_count = collection_articles.count()
    if existing_count > 0:
        print(f"\n Articles collection already exists with {existing_count} items")
        print(f"   New articles will be added (duplicates will be updated)")
    
    # Vectorize articles
    vectorize_articles(collection_articles, BOX_PATH, BOX_ARTICLES_FOLDER, use_markdown=True)
    
    print("\n" + "="*70)
    print(" Vectorization Complete!")
    print("="*70)
    
    # Show final stats
    articles_count = collection_articles.count()
    print(f" articles collection: {articles_count} items")
    
    # Show all collections
    all_collections = chroma_client.list_collections()
    if len(all_collections) > 1:
        print(f"\n All collections in database:")
        for c in all_collections:
            print(f"   {c.name}: {c.count()} items")
    
    print(f"\n Database location: {CHROMA_DB_DIR}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Vectorization interrupted by user")
    except Exception as e:
        print(f"\n\n Fatal error: {e}")
        import traceback
        traceback.print_exc()
        raise