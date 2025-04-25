cd /data/aculotta/transcribe_council
source v2/bin/activate
python transcribe_council.py
python summarize.py
python get_box_links.py
python vectorize.py
# zip up chroma db so it is accessible in box to fly.io app
cd box/chroma_db && tar czvf ../chroma_db.tar.gz * && cd ../../
