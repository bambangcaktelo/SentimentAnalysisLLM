import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

# --- Initialize Global Variables ---
client = None
db = None
users_collection = None
results_store_collection = None

# --- Establish Connection ---
if MONGODB_URI:
    try:
        # Create a new client and connect to the server
        client = AsyncIOMotorClient(MONGODB_URI)
        
        # Specify the database and collections
        db = client.get_database("sentiment_analysis_db")
        users_collection = db.get_collection("users")
        results_store_collection = db.get_collection("results")
        
        # The following line is for debugging and can be removed in production
        print("--- MongoDB connection successful. ---")

    except Exception as e:
        print(f"--- [ERROR] Could not connect to MongoDB: {e} ---")
        # If connection fails, variables will remain None, causing the error you saw
else:
    print("--- [WARNING] MONGODB_URI not found in .env file. Database functionality will be disabled. ---")
