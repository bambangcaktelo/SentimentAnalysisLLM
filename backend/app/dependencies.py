import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
import praw
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

# --- Load API Keys ---
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
REDDIT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# --- Initialize API Clients ---

# YouTube Client
youtube_client = None
if YOUTUBE_API_KEY:
    try:
        youtube_client = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        print("YouTube client initialized successfully.")
    except Exception as e:
        print(f"Error initializing YouTube client: {e}")
else:
    print("WARNING: YOUTUBE_API_KEY not found. YouTube scraping will be disabled.")

# Reddit Client
reddit_client = None
if REDDIT_ID and REDDIT_SECRET:
    try:
        reddit_client = praw.Reddit(
            client_id=REDDIT_ID,
            client_secret=REDDIT_SECRET,
            user_agent="SentimentAnalysisWebApp/1.0"
        )
        print("Reddit client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Reddit client: {e}")
else:
    print("WARNING: REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET not found. Reddit scraping will be disabled.")

# Hugging Face Router Client (for generative reports)
hf_client = None
if HUGGINGFACE_API_KEY:
    try:
        hf_client = AsyncOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HUGGINGFACE_API_KEY
        )
        print("Hugging Face client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Hugging Face client: {e}")
else:
    print("WARNING: HUGGINGFACE_API_KEY not found. LLM features will be disabled.")