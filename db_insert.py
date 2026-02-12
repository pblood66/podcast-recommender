## This script is used to insert data into the database
from glob import glob
import os
import json
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd

from utils import fast_pg_insert

load_dotenv()

CONNECTION = os.getenv("CONNECTION")

# Read the embedding files
def load_batch_requests(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            
            custom_id = obj["custom_id"]  
            podcast_id = obj["body"]['metadata']['podcast_id']  
            content = obj["body"]['input']
            start_time = obj["body"]['metadata']['start_time']  
            end_time = obj["body"]['metadata']['stop_time']
            rows.append({
                "custom_id": custom_id,
                "podcast_id": podcast_id,
                "content": content,
                "start_time": start_time,
                "stop_time": end_time
            })
    
    return pd.DataFrame(rows)

print("Loading batch request files...")
batch_request_files = glob("data/documents/*.jsonl")
batch_request_df = pd.concat(
    [load_batch_requests(file) for file in batch_request_files],
    ignore_index=True
)

# Read documents files
def load_embeddings(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            
            custom_id = obj["custom_id"]
            embedding = obj["response"]['body']["data"][0]["embedding"]
            
            rows.append({
                "custom_id": custom_id,
                "embedding": embedding
            })
    
    return pd.DataFrame(rows)

print("Loading embedding files...")
embedding_files = glob("data/embedding/*.jsonl")
embedding_dfs = pd.concat(
    [load_embeddings(file) for file in embedding_files],
    ignore_index=True
)

# Load the raw data via the hugging face datasets library
print("Loading HuggingFace dataset...")
ds = load_dataset("Whispering-GPT/lex-fridman-podcast")

hf_df = pd.DataFrame(ds['train'])
hf_df = hf_df.reset_index()
podcast_df = hf_df[["id", "title"]].drop_duplicates()

print(podcast_df.head())

# Merge batch requests with embeddings
print("\nMerging batch requests with embeddings...")
segment_df = batch_request_df.merge(
    embedding_dfs,
    left_on="custom_id",
    right_on="custom_id",
    how="inner"
)

segment_df = segment_df.rename(columns={
    "custom_id": "id"  
})

# Ensure correct data types
segment_df["start_time"] = segment_df["start_time"].astype(float)
segment_df["stop_time"] = segment_df["stop_time"].astype(float)

segment_df_final = segment_df[["id", "podcast_id", "content", "start_time", "stop_time", "embedding"]]

print(segment_df_final.head())


print('\n' + '='*50)
print('Inserting podcast data into the database...')
print('='*50)
fast_pg_insert(
    df=podcast_df,
    connection=CONNECTION,
    table_name="podcast",
    columns=["id", "title"]
)

print('\n' + '='*50)
print('Inserting segment data into the database...')
print('='*50)
fast_pg_insert(
    df=segment_df_final,
    connection=CONNECTION,
    table_name="segment",
    columns=["id", "podcast_id", "content", "start_time", "stop_time", "embedding"]
)
