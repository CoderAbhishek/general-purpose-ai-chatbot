import os
from dotenv import load_dotenv

load_dotenv()

# Load environment variables from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set.")

if not GROQ_MODEL:
    raise ValueError("GROQ_MODEL is not set.")