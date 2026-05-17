from firecrawl import Firecrawl
from langchain.tools import tool
from agent.utils.config import FIRECRAWL_API_KEY
firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)


# Scrape a website:
@tool
def scrape_website(website_link: str):
    """
        Extracts the full content of a single, specific webpage.
        Use this when you have a direct URL and need the markdown or HTML content
        to answer a specific question or perform data extraction from that page.

        Args:
            website_link (str): The exact URL of the webpage to be scraped.
        """

    scrape_status = firecrawl.scrape(
        website_link,
        formats=['markdown'],
        # This is the magic flag
    )

    return scrape_status

