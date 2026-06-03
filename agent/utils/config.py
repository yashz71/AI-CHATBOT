# agent/config.py
import os
from dotenv import load_dotenv

# This loads the variables from .env into the standard os.environ dictionary
load_dotenv(override=True)

# Extract variables safely
base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
base_url_em = os.getenv("BASE_URL_EM")
mcp_server_url = os.getenv("MCP_SERVER_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wealth_management_2026")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Fail-safe: Stop the server immediately if critical keys are missing
if not NEON_DATABASE_URL:
    raise ValueError("Critical configuration is missing! Check your .env file.")
