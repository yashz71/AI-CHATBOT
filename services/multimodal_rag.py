from llama_cloud import LlamaCloud
from agent.utils.config import NEON_DATABASE_URL, COLLECTION_NAME
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from agent.utils.config import base_url_em, api_key
from langchain_core.documents import Document as LCDocument
import time


# Initialize LlamaParse
client = LlamaCloud()  # reads LLAMA_CLOUD_API_KEY


embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=base_url_em,
    api_key=api_key,
    openai_api_version="2024-06-01",
)
# Initialize the store - it will NOT overwrite; it will just connect


async def ingest_pdf_to_pgvector(file_path: str, thread_id: str):

    # 1. Upload
    with open(file_path, "rb") as f:
        file = client.files.create(file=f, purpose="parse")

    # 2. Parse with polling
    result = client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"]
    )

    if not result.markdown or not result.markdown.pages:
        raise ValueError("No content extracted from PDF")

    raw_docs = []
    for page in result.markdown.pages:
        content = page.markdown or ""
        if not content.strip():
            continue

        raw_docs.append(LCDocument(
            page_content=content,
            metadata={"thread_id": thread_id, "page": page.page_number}
        ))
    # 4. Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='o200k_base',
        chunk_size=600,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(raw_docs)

    if not chunks:
        raise ValueError("No chunks generated")

    # 5. Store
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="users_docs",
        connection=NEON_DATABASE_URL,
        use_jsonb=True,
    )

    vectorstore.add_documents(chunks)

    return result

