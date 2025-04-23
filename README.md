# New Orleans City Council Video Downloader

This repo contains code to 
- Download N.O. City Council videos
- Transcribe them using Whisper
- Summarize them using Llama
- Vectorize them using ChromaDB

These scripts can be setup in a cronjob to be run daily, ensuring that the vector database contains the latest video transcripts.

- python transcribe_council.py
- python summarize.py
- python get_box_links.py
- python vectorize.py

## `transcribe_council.py`
 
- Fetches list of videos from https://cityofno.granicus.com/ViewPublisher.php?view_id=42
- Writes to `data.jsonl`
- For each video link:
  + Downloads video
  + Transcribes it with Whisper (version specified by `WHISPER_VERSION` in `.env`)
- Writes .mp4, .txt, and .json files

## `summarize.py`
- Summarizes all transcripts
- Uses Llama version specified in `LLAMA_VERSION` in `.env`
- Writes .summary files

## `get_box_links.py`

Gets shareable links to videos in Box and updates `data.jsonl` with new column `box_link`. Requires Box authentication (below)

## `vectorize.py`

Vectorize the summaries to a chroma vector database, stored in `CHROMA_DB_DIR`

### Box authentication

`get_box_links.py` requires Box API access. 

1. Create & configure your Box App
  - Go to Box Developer Console → Create New App
  - Select Custom App → OAuth 2.0 with OAuth 2.0 (User Authentication)
  - Under Configuration, set a Redirect URI such as http://127.0.0.1:5001/callback
  - Copy your Client ID and Client Secret to .env file
  - Under OAuth2 Scopes, enable write all files and folders stored in Box

2. Run python `box_app.py` once to authenticate your Box account and store tokens in `.env`
