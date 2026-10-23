import os

from dotenv import load_dotenv

# Load real values from .env first; load_dotenv() never overrides variables already set.
load_dotenv()

# Dummy values only fill gaps (e.g. a machine without .env), so imports of config.py do not fail.
os.environ.setdefault("GROQ_API_KEY", "test-api-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("GROQ_MODEL", "test-model")
