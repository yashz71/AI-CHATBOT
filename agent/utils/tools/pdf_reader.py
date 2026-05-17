from llama_cloud import LlamaCloud
from langchain.tools import tool


@tool
def read_pdf(pdf: str):
    client = LlamaCloud()  # reads LLAMA_CLOUD_API_KEY
    file = client.files.create(file=pdf, purpose="parse")
    result = client.parsing.parse(file_id=file.id, tier="agentic", version="latest", expand=['markdown'])
    print(result.markdown.pages[0].markdown)
