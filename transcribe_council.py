"""
Transcribes each video in BOX_PATH/.mp4 using Whisper (version determined by WHISPER_VERSION).
Output written to BOX_PATH/.txt (raw text output) and BOX_PATH/.json (time stamped chunks).
List of links stored in BOX_PATH/data.json.

Skips any videos that already have a corresponding .json transcript file.
"""
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import os
import pandas as pd
import re
import requests
import time
import whisper

load_dotenv()  # Loads variables from .env into environment
URL = os.getenv('COUNCIL_VIDEO_URL')
PATH = os.getenv('BOX_PATH')

def get_date_time(raw_text):
    match = re.search(r"(\w+,\s\w+\s\d{1,2},\s\d{4})\s*-\s*(\d{1,2}:\d{2}\s*[APMapm]{2})", raw_text)
    if match:
        return match.group(1), match.group(2)
    return raw_text, "Unknown Time"

def get_all_links():
    """
    get all links to videos, agendas, minutes
    """
    response = requests.get(URL)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    meetings = []
    rows = soup.find_all("tr", class_="listingRow")

    for row in rows:
        meeting_data = {}

        columns = row.find_all("td", class_="listItem")

        if len(columns) >= 2:
            meeting_data["title"] = columns[0].get_text(strip=True)

            raw_date_time = ' '.join(columns[1].get_text(separator=' ', strip=True).split())
            meeting_data["date"], meeting_data["time"] = get_date_time(raw_date_time)

            for a in row.find_all("a", href=True):
                href = a["href"].strip()

                if href == "javascript:void(0);" and "onclick" in a.attrs:
                    onclick_text = a["onclick"]
                    match = re.search(r"window\.open\('([^']+)'", onclick_text)
                    if match:
                        video_page_url = "https:" + match.group(1)  # Ensure it's a full URL
                        meeting_data["video_page"] = video_page_url
    
                if href.startswith("//"):
                    href = "https:" + href
                if ".mp4" in href:
                    meeting_data["video"] = href
                elif "AgendaViewer.php" in href:
                    meeting_data["agenda"] = href
                elif "MinutesViewer.php" in href:
                    meeting_data["minutes"] = href

            if "video" in meeting_data:
                meetings.append(meeting_data)

    return pd.DataFrame(meetings)


def dl_video(url, fname):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(fname, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print('downloaded %s to %s' % (url, fname))
    time.sleep(2)


def transcribe_video(fname, model, txt_fname, json_fname):
    result = model.transcribe(fname, verbose=False)
    open(txt_fname, 'wt').write(result['text'] + '\n')
    pd.DataFrame(result['segments']).to_json(json_fname, orient='records', lines=True)
    print('saved to %s and %s' % (txt_fname, json_fname))

def cp_box_links(newdf):
    dataf = PATH + 'data.jsonl'
    # if data.jsonl already exists.
    if os.path.exists(dataf):
        olddf = pd.read_json(dataf, orient='records', lines=True)
        video2box = dict(olddf[['video', 'box_link']].values)
        newdf['box_link'] = [video2box.get(v, None) for v in newdf.video]

os.makedirs(PATH, exist_ok=True)
# get all links.
print('fetching links...')
df = get_all_links()
cp_box_links(df)
# save to file
df.to_json(PATH + 'data.jsonl', orient='records', lines=True)
print('saved to data.jsonl')

# load whisper
whisper_model = whisper.load_model(os.getenv('WHISPER_VERSION'),
                                   device=os.getenv('GPU_DEVICE', 'cpu'))

# process all files, skipping those already done
for _, row in df.iterrows():
    print(row.title, row.date)
    if row.video is not None:
        fname = PATH + os.path.basename(row.video)
        # dl video
        if not os.path.exists(fname):
            print('downloading %s' % row.video)
            dl_video(row.video, fname)
        else:
            print('got it')
        # transcribe video
        txt_fname = re.sub('.mp4', '.txt', fname)
        json_fname = re.sub('.mp4', '.json', fname)
        if not os.path.exists(txt_fname) or not os.path.exists(json_fname):
            print('transcribing %s' % fname)
            transcribe_video(fname, whisper_model, txt_fname, json_fname)
        else:
            print('already transcribed %s' % fname)
        
