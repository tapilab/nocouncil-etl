"""
Vectorize the .summary files from BOX_PATH and save to a chroma database in CHROMA_DB_DIR

We use by default a simple embedding -- all-MiniLM-L6-v2 (384 dim) -- so that we can use the same
on a cheap fly.io server to embed the query.
"""
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function \
    import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
import glob
import json
import numpy as np
# import ollama
import os
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm

load_dotenv()  # Loads variables from .env into environment
PATH = os.getenv('BOX_PATH')

# unused: optionally embed using models in ollama server
# class MyEmbeddingFunction(EmbeddingFunction[Documents]):
#     """
#     This class is used to get embeddings for a list of texts using Ollama Python Library.
#     It requires a host url and a model name. The default model name is "nomic-embed-text".
#     """

#     def __init__(
#         self, host: str = "http://localhost:11434", model_name: str = "snowflake-arctic-embed2"
#     ):
#         self._client = ollama.Client(host=host)
#         self._model_name = model_name

#     # "nomic-embed-text"
#     def __call__(self, input: Documents) -> Embeddings:
#         """
#         Get the embeddings for a list of texts.
#         Args:
#             input (Documents): A list of texts to get embeddings for.
#         Returns:
#             Embeddings: The embeddings for the texts.
#         Example:
#             >>> ollama = OllamaEmbeddingFunction(host="http://localhost:11434")
#             >>> texts = ["Hello, world!", "How are you?"]
#             >>> embeddings = ollama(texts)
#         """

#         embeddings = []
#         # Call Ollama Embedding API for each document.
#         for document in input:
#             embedding = self._client.embeddings(model=self._model_name, prompt=document)
#             embeddings.append(embedding["embedding"])
#         return embeddings

def filename2date(filename, df):
    mp4 = re.sub('.summary', '.mp4', filename.split('/')[-1])
    return df[df.video.str.contains(mp4)].iloc[0].date

def make_vector_db(collection, file_iter):
    """
    Make the vector database and add to chroma collection.
    collection....chroma collection
    file_iter.....iterator over .summary files to be vectorized.
    """
    df = pd.read_json(PATH + 'data.jsonl', orient='records', lines=True)
    print('embedding summaries...')
    files = list(file_iter)
    for sfile in tqdm(files):
        jsons = [json.loads(l) for l in open(sfile)][1:]
        jsons = [j for j in jsons if len(j['summary'].strip()) > 0]
        # drop first summary as it covers the full meeting
        # drop empty summaries
        if len(jsons) == 0:
            continue
        fdate = None
        try:
            fdate = filename2date(sfile, df)
        except Exception as e:
            print('cannot parse %s' % sfile)
            continue
        # flatten quotes and names into single strings, per chroma
        for i in range(len(jsons)):
            j = jsons[i]
            j['quotes'] = '|||'.join(j['quotes'])
            j['names'] = '|||'.join(j['names'])

        metas = [{k:v for k,v in j.items() if k!='summary'} | 
                       {'file':sfile, 'date': int(fdate.timestamp())} for j in jsons]
        collection.add(
            documents=[j['summary'] for j in jsons],
            metadatas=metas,
            ids=[sfile + ':' + str(j['start_id']) + ':' + str(j['end_id']) for j in jsons]
        )
    print('...done')
    return collection


if __name__ == "__main__":
    load_dotenv()  # load OPENAI_API_KEY, CHROMA_DB_DIR
    DB_DIR = os.getenv("CHROMA_DB_DIR", 'chroma_db')
    # ─── Initialize ChromaDB
    chroma_client = PersistentClient(
        path=DB_DIR,            # where on disk to store
        settings=Settings(anonymized_telemetry=False)
    )

    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",  # small, fast, 384‑dim
        device="cpu",                   # or "cuda"
        normalize_embeddings=True
    )
    collection = chroma_client.get_or_create_collection(
        name="city_council",
        embedding_function=embed_fn, 
        metadata={"hnsw:space": "cosine",
                  "hnsw:num_threads": 1})

    make_vector_db(collection, glob.glob(os.getenv('BOX_PATH') + '*.summary'))
