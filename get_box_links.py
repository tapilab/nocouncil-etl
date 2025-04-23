"""
- Iterates through all mp4 files in BOX_PATH folder and
makes public links if needed.
- Saves resulting link to the 'box_link' column of BOX_PATH/data.jsonl.
- Requires BOX API credentials in .env. (run box_app.py prior on first run)
"""
from boxsdk import OAuth2, Client
from dotenv import load_dotenv
import os
import pandas as pd
import re
from tqdm import tqdm

load_dotenv()  # Loads variables from .env into environment
PATH = os.getenv('BOX_PATH')

def store_tokens(access_token, refresh_token):
    # overwrite the existing lines in .env
    lines = []
    with open(".env") as f:
        for line in f:
            if line.startswith("BOX_ACCESS_TOKEN") or line.startswith("BOX_REFRESH_TOKEN"):
                continue
            lines.append(line)
    with open(".env", "w") as f:
        f.writelines(lines)
        f.write(f"BOX_ACCESS_TOKEN={access_token}\n")
        f.write(f"BOX_REFRESH_TOKEN={refresh_token}\n")

auth = OAuth2(
    client_id=os.getenv("BOX_CLIENT_ID"),
    client_secret=os.getenv("BOX_CLIENT_SECRET"),
    access_token=os.getenv("BOX_ACCESS_TOKEN"),
    refresh_token=os.getenv("BOX_REFRESH_TOKEN"),
    store_tokens=store_tokens,  # this callback gets invoked on every refresh
)

client = Client(auth)
folder_id = os.getenv('BOX_FOLDER_ID')
folder = client.folder(folder_id).get()
items = folder.get_items(limit=1000)
df = pd.read_json(PATH + 'data.jsonl', orient='records', lines=True)
mp42box = {}
if 'box_link' not in df.columns:
    df['box_link'] = None

for item in tqdm(items):
    if item.type == 'file' and item.name.endswith('.mp4'):
        if pd.isnull(df[df.video.str.contains(item.name)].box_link.iloc[0]):
            full_item = client.file(item.id).get()  # Fetch full metadata

            # Check for existing shared link
            # if not getattr(full_item, 'shared_link', None):
            shared_link = full_item.get_shared_link(access='open', allow_download=True)
            # else:
                # shared_link = full_item.shared_link['url']
            # if has /s/, need to change to static
            if '/s/' in shared_link:
                shared_link = re.sub('/s/', '/shared/static/', shared_link) + '.mp4?dl=1'
            mp42box[item.name] = shared_link

def find_box_link(r, mp42box):
    if not pd.isnull(r.box_link):
        return r.box_link
    mp4 = r.video.split('/')[-1]
    return mp42box.get(mp4, None)

df['box_link'] = df.apply(find_box_link, args=(mp42box,), axis=1)
df.to_json(PATH + 'data.jsonl', orient='records', lines=True)