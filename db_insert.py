## This script is used to insert data into the database
from glob import glob
import os
import json
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd
import psycopg2

# from recommender.utils import fast_pg_insert

load_dotenv()

CONNECTION = os.getenv("CONNECTION_STRING")

# TODO: Read the embedding files
def load_batch_requests(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            
            custom_id = obj["custom_id"]  # "0;1"
            podcast_id = obj["body"]['metadata']['podcast_id']  # "0"
            content = obj["body"]['input']
            start_time = obj["body"]['metadata']['start_time']  # transcription text
            end_time = obj["body"]['metadata']['stop_time']
            rows.append({
                "custom_id": custom_id,
                "podcast_id": podcast_id,
                "content": content,
                "start_time": start_time,
                "end_time": end_time
            })
    
    return pd.DataFrame(rows)

batch_request_files = glob("data/documents/*.jsonl")
batch_request_df = pd.concat(
    [load_batch_requests(file) for file in batch_request_files],
    ignore_index=True
)
print(batch_request_df.head().to_string())

# TODO: Read documents files
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

embedding_files = glob("data/embedding/*.jsonl")
embedding_dfs = pd.concat(
    [load_embeddings(file) for file in embedding_files],
    ignore_index=True
)

print(embedding_dfs.head().to_string())
# HINT: In addition to the embedding and document files you likely need to load the raw data via the hugging face datasets library
ds = load_dataset("Whispering-GPT/lex-fridman-podcast")
print(f"Loaded {len(ds['train'])} podcast episodes")

hf_df = pd.DataFrame(ds['train'])
hf_df = hf_df.reset_index()
podcast_df = hf_df[["id", "title"]].drop_duplicates()

print(podcast_df.head())


segment_df = batch_request_df.merge(
    embedding_dfs,
    left_on="custom_id",
    right_on="custom_id",
    how="inner"
)

segment_df = segment_df.drop(columns=["custom_id"])
print(segment_df.head())


# TODO: Insert into postgres
# HINT: use the recommender.utils.fast_pg_insert function to insert data into the database
# otherwise inserting the 800k documents will take a very, very long time
# conn = psycopg2.connect(CONNECTION);



