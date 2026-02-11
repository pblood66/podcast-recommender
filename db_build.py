## This script is used to create the tables in the database

import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

CONNECTION = None # paste connection string here or read from .env file

# need to run this to enable vector data type
CREATE_EXTENSION = "CREATE EXTENSION vector"

# TODO: Add create table statement
CREATE_PODCAST_TABLE = """

"""
# TODO: Add create table statement
CREATE_SEGMENT_TABLE = """

"""

conn = psycopg2.connect(CONNECTION)
# TODO: Create tables with psycopg2 (example: https://www.geeksforgeeks.org/executing-sql-query-with-psycopg2-in-python/)


