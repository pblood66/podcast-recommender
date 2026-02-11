## This script is used to query the database
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

CONNECTION = None # paste connection string here or read from .env file

