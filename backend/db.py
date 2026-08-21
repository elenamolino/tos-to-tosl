"""MongoDB connection for persisting pipeline jobs."""

import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "tos_to_odrl")

_client = MongoClient(MONGODB_URI)
_db = _client[MONGODB_DB]

jobs_collection = _db["jobs"]
jobs_collection.create_index("job_id", unique=True)
