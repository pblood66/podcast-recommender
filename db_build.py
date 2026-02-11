## This script is used to create the tables in the database

import os
from dotenv import load_dotenv, find_dotenv
import psycopg2

_ = load_dotenv(find_dotenv())
CONNECTION=os.getenv('CONNECTION')

print(CONNECTION)

# need to run this to enable vector data type
CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

# TODO: Add create table statement
CREATE_PODCAST_TABLE = """
    CREATE TABLE IF NOT EXISTS podcast(
        id VARCHAR(255) PRIMARY KEY,
        TITLE TEXT NOT NULL
    );
"""
# TODO: Add create table statement
CREATE_SEGMENT_TABLE = """
    CREATE TABLE IF NOT EXISTS segment(
        id VARCHAR(255) PRIMARY KEY,
        start_time FLOAT NOT NULL,
        stop_time FLOAT NOT NULL,
        content TEXT,
        embedding VECTOR(128),
        FOREIGN KEY(id) REFERENCES podcast(id) 
            ON DELETE CASCADE
    )
"""

conn = psycopg2.connect(CONNECTION)
cur = conn.cursor()

try:
    # Enable vector extension
    print("Enabling pgvector extension...")
    cur.execute(CREATE_EXTENSION)
    conn.commit()
    print("pgvector extension enabled")
    
    # Create podcast table
    print("Creating podcast table...")
    cur.execute(CREATE_PODCAST_TABLE)
    conn.commit()
    print("podcast table created")
    
    # Create podcast_segment table
    print("Creating podcast_segment table...")
    cur.execute(CREATE_SEGMENT_TABLE)
    conn.commit()
    print("podcast_segment table created")
    
    print("\nAll tables created successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
    
finally:
    cur.close()
    conn.close()
    print("Database connection closed.")

