from agent.utils.config import NEON_DATABASE_URL, COLLECTION_NAME
from langchain_postgres.vectorstores import PGVector
from langchain.tools import tool
from langchain_openai import AzureOpenAIEmbeddings
from agent.utils.config import base_url_em, api_key


embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=base_url_em,
    api_key=api_key,
    openai_api_version="2024-06-01",
)
# Initialize the store - it will NOT overwrite; it will just connect
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=NEON_DATABASE_URL,
    use_jsonb=True,
)


@tool
def finance_knowledge_base(query: str) -> str:
    """
    Consult the internal 'Wealth Management 2026' reports for specialized
    macro-economic data, investment outlooks, and institutional insights.
    Use this for deep financial strategy questions.
    """
    # 1. Retrieve the most relevant chunks from Neon
    # We limit to k=3 to keep the context window small for your 2012 Mac
    context_docs = vector_store.similarity_search(query, k=5)
    print(f"DEBUG: Found {len(context_docs)} chunks in DB for query: {query}")
    # 2. Format the results clearly for the LLM
    # Using a list comprehension is memory-efficient
    context_text = "\n\n".join([
        f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}"
        for d in context_docs
    ])

    if not context_text:
        return "No specific information found in the internal reports."
    print(f"test: {context_text}")
    return f"Extracted Findings from Internal Reports:\n{context_text}"
