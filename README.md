# New Orleans City Council Video Downloader

This repo contains code to 
- Download N.O. City Council videos
- Transcribe them using Whisper
- Correct known transcription errors (streets, names) using a fuzzy-matching pipeline
- Summarize them using Llama
- Vectorize them using ChromaDB
- Crawl New Orleans civic news RSS feeds and save relevant articles to Box
- Vectorize those news articles into a separate ChromaDB collection

## Requirements

These scripts depend on **Python 3.9** (pinned in `requirements.txt`, e.g. `chromadb==1.0.5` and the `openai-whisper` git dependency). Use a Python 3.9 virtualenv — other versions are not tested and may fail to install or run correctly.

These scripts can be setup in a cronjob to be run regularly, ensuring that the vector database contains the latest video transcripts.

- python transcribe_council.py
- python correct_transcript.py
- python summarize.py
- python get_box_links.py
- python vectorize.py
- python newscrawler.py
- python article_vectorize.py

## `transcribe_council.py`
 
- Fetches list of videos from https://cityofno.granicus.com/ViewPublisher.php?view_id=42
- Writes to `data.jsonl`
- For each video link:
  + Downloads video
  + Transcribes it with Whisper (version specified by `WHISPER_VERSION` in `.env`)
- Writes .mp4, .txt, and .json files

## `correct_transcript.py`

- Runs the `correction.py` pipeline over each `.json` transcript in `BOX_PATH`
- Concatenates a transcript's segments into one string (for full-document context), corrects it, then redistributes the corrected words back into their original timestamped segments
- Writes `.corrected.json` alongside the original `.json`, leaving the raw transcript untouched
- Skips transcripts that already have a `.corrected.json`

## `correction.py`

The correction engine used by `correct_transcript.py`. Deterministic, fuzzy-matching only — no LLM calls. Runs three stages in order over the transcript text:

- **Stage 0 — hardcoded fixes**: exact (case-insensitive) find/replace for known, frequent Whisper mistranscriptions, loaded from `hardcoded_corrections.json` and merged with a few built-in defaults
- **Stage 1 — street names**: fuzzy-matches 1–3 word spans preceding a street-type word (e.g. "street", "ave.") against `nola_streets.json`, boosting scores for prefix-truncation typos, and skipping spans that are common English words or look like a person's name
- **Stage 2 — person names**: fuzzy-matches the word following a title (e.g. "councilmember", "mayor", "dr.") against the name list in `nola_names.json`, using document-frequency counts to boost confidence when the corrected spelling already appears elsewhere in the transcript
- Exposes `load_dictionaries()` and `correct_transcript()`, which `correct_transcript.py` imports directly

### Correction dictionaries

- **`nola_names.json`** — `{"first_names": [...], "last_names": [...]}`, New Orleans-area council/community first and last names used by Stage 2 name correction
- **`nola_streets.json`** — flat list of ~2,600 full New Orleans street names (e.g. `"Canal Street"`) used by Stage 1 street correction
- **`english_words.json`** — flat list of ~236k common English words; used to avoid "correcting" ordinary words that happen to fuzzy-match a street or name
- **`hardcoded_corrections.json`** — small `{"misspelling": "Correct"}` map of known recurring Whisper errors (e.g. council member names Whisper consistently mangles), applied verbatim before the fuzzy stages run

## `summarize.py`
- Summarizes all corrected transcripts (`.corrected.json`)
- Uses Llama version specified in `LLAMA_VERSION` in `.env`
- Writes .summary files

## `get_box_links.py`

Gets shareable links to videos in Box and updates `data.jsonl` with new column `box_link`. Requires Box authentication (below)

## `vectorize.py`

Vectorize the summaries to a chroma vector database, stored in `CHROMA_DB_DIR`

## `newscrawler.py`

- Fetches New Orleans civic news RSS feeds (Verite News, The Lens, New Orleans CityBusiness)
- Extracts full article text for each new entry (tries `trafilatura` first, falls back to a headless Selenium/Chrome fetch)
- Filters articles to those matching a civic-relevance keyword list (city council, zoning, budget, etc.)
- Saves each relevant article as a Markdown file (with frontmatter) into the Box-mounted `BOX_ARTICLES_FOLDER`, and tracks all saved articles in `articles.json` in that same folder
- Skips URLs already recorded in `articles.json`
- Requires a mounted Box folder — set `BOX_PATH` (and optionally `BOX_ARTICLES_FOLDER`, default `news`) in `.env`

## `article_vectorize.py`

- Loads articles from `BOX_PATH/BOX_ARTICLES_FOLDER` (both `articles.json` and the individual `.md` files written by `newscrawler.py`), dedupes them, and chunks each article on paragraph/sentence boundaries
- Adds/upserts the chunks into an `articles` collection in the existing ChromaDB at `CHROMA_DB_DIR` (the same database used by `vectorize.py`, kept as a separate collection)
- Requires `CHROMA_DB_DIR`, `BOX_PATH`, and optionally `BOX_ARTICLES_FOLDER` (default `articles`) in `.env`

### Box authentication

`get_box_links.py` requires Box API access. 

1. Create & configure your Box App
  - Go to Box Developer Console → Create New App
  - Select Custom App → OAuth 2.0 with OAuth 2.0 (User Authentication)
  - Under Configuration, set a Redirect URI such as http://127.0.0.1:5001/callback
  - Copy your Client ID and Client Secret to .env file
  - Under OAuth2 Scopes, enable write all files and folders stored in Box

2. Run python `box_app.py` once to authenticate your Box account and store tokens in `.env`
