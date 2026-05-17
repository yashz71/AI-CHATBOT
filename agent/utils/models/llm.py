from langchain_openai import ChatOpenAI
from agent.utils.config import base_url, api_key
from agent.utils.tools.firecrawl import scrape_website

model = ChatOpenAI(
    model="gpt-4o-mini",
    base_url=base_url,
    api_key=api_key,
    temperature=0,
    stream_usage=True,
    max_tokens=None,
    timeout=None,
    max_retries=2
)
tools = [scrape_website]

scrape_model_with_tools = model.bind_tools(tools)
