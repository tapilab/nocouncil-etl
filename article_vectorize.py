"""
Articles Vectorizer
Adds news articles collection to existing ChromaDB
Reads from Box-mounted folders, writes to local ChromaDB
"""
import os
import re
import warnings

# Disable ChromaDB telemetry BEFORE importing chromadb
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Suppress ChromaDB telemetry warnings
warnings.filterwarnings('ignore', message='.*telemetry.*')

import chromadb
from chromadb import PersistentClient
from chromadb.api.client import Client as ClientCreator
from chromadb.config import Settings
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function \
    import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
import email.utils
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


def make_persistent_chroma_client(chroma_db_dir: str):
    """
    Create a persistent Chroma client.

    Chroma 1.0.5 can panic inside the Rust-backed PersistentClient on some
    existing local databases (e.g. one whose sqlite schema was forward-migrated
    by a newer chromadb version). When that happens, fall back to the older
    SegmentAPI implementation, which is slower but more robust for this repo's
    existing on-disk state.
    """
    base_settings = Settings(anonymized_telemetry=False)
    chroma_version = getattr(chromadb, "__version__", "")
    force_segment = os.getenv("CHROMA_FORCE_SEGMENT_API", "").lower() in {"1", "true", "yes"}

    if force_segment:
        print("   Using SegmentAPI persistent client (env override)")
        fallback_settings = Settings(anonymized_telemetry=False)
        fallback_settings.chroma_api_impl = "chromadb.api.segment.SegmentAPI"
        fallback_settings.is_persistent = True
        fallback_settings.persist_directory = chroma_db_dir
        return ClientCreator(settings=fallback_settings)

    try:
        return PersistentClient(
            path=chroma_db_dir,
            settings=base_settings,
        )
    except BaseException as exc:
        print("   Rust PersistentClient failed, retrying with SegmentAPI fallback...")
        print(f"   Reason: {type(exc).__name__}: {exc}")

        fallback_settings = Settings(anonymized_telemetry=False)
        fallback_settings.chroma_api_impl = "chromadb.api.segment.SegmentAPI"
        fallback_settings.is_persistent = True
        fallback_settings.persist_directory = chroma_db_dir

        return ClientCreator(settings=fallback_settings)


def parse_published_to_unix(published: str) -> Optional[int]:
    """
    Parse a published date string in RFC 2822 format to a Unix timestamp (int).
    Handles formats like: 'Tue, 02 Sep 2025 20:57:54 +0000'

    Returns Unix timestamp as int, or None if parsing fails.
    """
    if not published:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(published)
        return int(parsed.timestamp())
    except Exception:
        return None


def split_into_sentences(text: str) -> list[str]:
    """Split text into approximate sentences without external dependencies."""
    normalized = re.sub(r'\s+', ' ', text).strip()
    if not normalized:
        return []

    # Keep sentence-ending punctuation attached to the sentence.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', normalized)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_semantically(
    text: str,
    target_words: int = 350,
    max_words: int = 450,
    overlap_sentences: int = 1,
) -> list[str]:
    """
    Chunk text on paragraph/sentence boundaries rather than raw word windows.

    Strategy:
    - preserve paragraph boundaries when possible
    - accumulate full sentences up to a target size
    - cap chunk size to avoid giant passages
    - carry a small sentence overlap for retrieval continuity
    """
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_words = 0
    pending_new_sentences = 0

    def flush_chunk() -> None:
        nonlocal current_sentences, current_words, pending_new_sentences
        if not current_sentences or pending_new_sentences == 0:
            return
        chunks.append(' '.join(current_sentences).strip())
        overlap = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
        current_sentences = overlap[:]
        current_words = sum(len(s.split()) for s in current_sentences)
        pending_new_sentences = 0

    for paragraph in paragraphs:
        sentences = split_into_sentences(paragraph)
        if not sentences:
            continue

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # Split unusually long sentences so one sentence cannot dominate a chunk.
            if sentence_words > max_words:
                if current_sentences:
                    flush_chunk()
                words = sentence.split()
                for i in range(0, len(words), target_words):
                    piece = ' '.join(words[i:i + target_words]).strip()
                    if piece:
                        chunks.append(piece)
                current_sentences = []
                current_words = 0
                pending_new_sentences = 0
                continue

            would_exceed_max = current_words > 0 and current_words + sentence_words > max_words
            reached_target = current_words >= target_words
            if would_exceed_max or reached_target:
                flush_chunk()

            current_sentences.append(sentence)
            current_words += sentence_words
            pending_new_sentences += 1

        # Prefer to end chunks at paragraph boundaries once they are reasonably full.
        if current_words >= target_words:
            flush_chunk()

    if current_sentences and pending_new_sentences > 0:
        chunks.append(' '.join(current_sentences).strip())

    # Drop accidental duplicates from overlap-only flushes.
    deduped_chunks: list[str] = []
    for chunk in chunks:
        if chunk and (not deduped_chunks or deduped_chunks[-1] != chunk):
            deduped_chunks.append(chunk)
    return deduped_chunks


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


def parse_markdown_frontmatter(md_path: Path) -> Tuple[Dict, str]:
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
    
    for md_path in tqdm(md_files, desc="Loading markdown", unit="file"):
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


def vectorize_articles(
    collection,
    box_path: str,
    articles_folder: str,
    use_markdown: bool = True,
    target_chunk_words: int = 350,
    max_chunk_words: int = 450,
    overlap_sentences: int = 1,
):
    """
    Vectorize news articles from articles.json and/or markdown files.
    Each article is split into overlapping chunks so retrieval returns
    the relevant passage rather than the full article text.

    Args:
        collection: ChromaDB collection for articles
        box_path: Path to Box root folder
        articles_folder: Relative folder name or full path to articles
        use_markdown: If True, also load from individual .md files
        target_chunk_words: Preferred chunk size in words
        max_chunk_words: Hard cap for chunk size in words
        overlap_sentences: Number of trailing sentences to repeat between chunks
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

    # Deduplicate repeated articles loaded from JSON + markdown.
    deduped_articles = []
    seen_article_keys = set()
    for article in articles:
        article_key = (
            article.get('id')
            or article.get('url')
            or article.get('filename')
            or f"{article.get('title', '')}:{len(article.get('content', ''))}"
        )
        if article_key in seen_article_keys:
            continue
        seen_article_keys.add(article_key)
        deduped_articles.append(article)
    articles = deduped_articles
    
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
        
        # Parse published date to Unix timestamp
        published_str = article.get('published', '')
        published_unix = parse_published_to_unix(published_str)

        # Build base metadata shared across all chunks of this article
        base_metadata = {
            'title': article.get('title', ''),
            'url': article.get('url', ''),
            'source': article.get('source', ''),
            'published': published_str,
            'published_unix': published_unix,
            'saved_at': article.get('saved_at', ''),
            'filename': article.get('filename', ''),
        }
        # Remove empty string values (but keep 0 and valid ints)
        base_metadata = {k: v for k, v in base_metadata.items() if v is not None and v != ''}

        # Split article into semantically aligned chunks.
        article_id = article.get('id') or article.get('url') or f"article_{i}"
        chunks = chunk_text_semantically(
            content,
            target_words=target_chunk_words,
            max_words=max_chunk_words,
            overlap_sentences=overlap_sentences,
        )

        for j, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                **base_metadata,
                'chunk': j,
                'total_chunks': len(chunks),
                'chunking_strategy': 'semantic_sentence_paragraph',
            })
            ids.append(f"{article_id}_chunk{j}")
    
    # Add to collection in batches
    print('🔄 Adding to ChromaDB...')
    batch_size = 500
    total_added = 0
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        collection.upsert(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        
        total_added += len(batch_docs)
    
    print(f'Done! Added {total_added} chunks from {len(articles)} articles')
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
    chroma_client = make_persistent_chroma_client(CHROMA_DB_DIR)
    print(f"   Using API implementation: {chroma_client.get_settings().chroma_api_impl}")
    
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
        print(f"   Existing article chunks will be upserted by stable IDs")
    
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
