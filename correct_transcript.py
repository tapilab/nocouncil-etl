"""
Correct transcripts in BOX_PATH/.json using the three-stage fuzzy
correction pipeline (hardcoded → streets → names).

Reads each .json transcript (JSONL format with timestamped segments),
corrects the 'text' field in every segment, and writes the result to
BOX_PATH/.corrected.json.  The original .json is left untouched.

summarize.py should then be pointed at the .corrected.json files
instead of the raw .json files.

Skips any transcripts that already have a corresponding .corrected.json.

Required files in the same directory as this script:
  - correction.py             (the correction engine)
  - english_words.json
  - nola_names.json
  - nola_streets.json
  - hardcoded_corrections.json
"""

from dotenv import load_dotenv
import json
import os
import re

from correction import load_dictionaries, correct_transcript

load_dotenv()
PATH = os.getenv('BOX_PATH')

# Subfolder for corrected transcripts (created automatically)
CORRECTED_FOLDER = os.path.join(PATH, 'corrected_transcripts/')
os.makedirs(CORRECTED_FOLDER, exist_ok=True)

# ── Load dictionaries once ───────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
dicts = load_dictionaries(
    english_path=os.path.join(script_dir, "english_words.json"),
    names_path=os.path.join(script_dir, "nola_names.json"),
    streets_path=os.path.join(script_dir, "nola_streets.json"),
)


def correct_json_transcript(json_fname, corrected_fname):
    """
    Read a JSONL transcript file, correct the 'text' field in each
    segment, and write the corrected version to corrected_fname.

    All other fields (id, start, end, no_speech_prob, etc.) are
    preserved exactly as-is.
    """
    # Read all segments
    segments = []
    with open(json_fname, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                segments.append(json.loads(line))

    if len(segments) == 0:
        print('  skipping empty transcript')
        return 0

    # Combine all text for correction (gives the algorithm full
    # document context for frequency-based confidence boosting)
    full_text = ' '.join(seg.get('text', '') for seg in segments)
    corrected_text, corrections = correct_transcript(full_text, dicts)

    if len(corrections) == 0:
        # No corrections needed — still write the file so we skip it
        # next time, but segments are unchanged.
        print('  no corrections needed')
    else:
        # Map corrections back to individual segments.
        # We corrected the concatenated text, so now we need to
        # redistribute the corrected words back to each segment
        # based on their original word counts.
        corrected_words = corrected_text.split()
        word_idx = 0

        for seg in segments:
            orig_word_count = len(seg.get('text', '').split())
            seg_words = corrected_words[word_idx:word_idx + orig_word_count]
            # Preserve leading space if the original had one (Whisper
            # segments typically start with a space)
            if seg.get('text', '').startswith(' ') and seg_words:
                seg['text'] = ' ' + ' '.join(seg_words)
            else:
                seg['text'] = ' '.join(seg_words)
            word_idx += orig_word_count

    # Write corrected JSONL
    with open(corrected_fname, 'w', encoding='utf-8') as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + '\n')

    print('  saved to %s (%d corrections)' % (corrected_fname, len(corrections)))
    return len(corrections)


# ── Main loop — mirrors the pattern in transcribe_council.py ─────
import pandas as pd

# Set to None to process ALL files, or a number to limit (e.g., 5)
LIMIT = None

df = pd.read_json(PATH + 'data.jsonl', orient='records', lines=True)

total_files = 0
total_corrections = 0
skipped = 0

for _, row in df.iterrows():
    if LIMIT is not None and total_files >= LIMIT:
        print('\n  Reached limit of %d files, stopping.' % LIMIT)
        break

    print(row.title, row.date)
    if row.video is not None:
        fname = PATH + os.path.basename(row.video)
        json_fname = re.sub('.mp4', '.json', fname)
        corrected_fname = os.path.join(
            CORRECTED_FOLDER, os.path.basename(re.sub('.mp4', '.corrected.json', fname))
        )

        if not os.path.exists(json_fname):
            print('  no transcript found, skipping')
            continue

        if os.path.exists(corrected_fname):
            print('  already corrected')
            skipped += 1
            continue

        print('  correcting %s' % json_fname)
        n = correct_json_transcript(json_fname, corrected_fname)
        total_files += 1
        total_corrections += n

print('\n' + '=' * 60)
print('DONE')
print('  Files corrected: %d' % total_files)
print('  Files skipped:   %d' % skipped)
print('  Total corrections: %d' % total_corrections)
print('=' * 60)