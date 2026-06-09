# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
import shutil
from fastapi.responses import FileResponse
import os
import uuid
from pydantic import BaseModel
from agent.utils.historical_subgraph.graph import build_general_subgraph
from agent.utils.general_subgraph.workflow import build_finance_agent
from services.multimodal_rag import ingest_pdf_to_pgvector
from typing import Optional
from agent.startup.mcp import initialize_mcp

# LangGraph Imports
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from agent.utils.config import NEON_DATABASE_URL
from agent.utils.workflow import build_workflow  # Import your StateGraph

# --- Global Graph Instance ---
# We define this globally so the API routes can access the compiled graph
agent_app = None


# --- Lifespan: The "Mac-Friendly" Connection Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_app
    # 1. Initialize the Async Checkpointer (Persistent Memory)
    # Using 'autocommit=True' is a 2026 requirement for Neon/Postgres
    deps = await initialize_mcp()
    subgraph = build_general_subgraph(
        deps["tool_node"]
    )
    finance_agent = build_finance_agent()

    workflow = build_workflow(
        subgraph, finance_agent
    )
    langgraph_db_url = NEON_DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    async with AsyncPostgresSaver.from_conn_string(langgraph_db_url) as saver:
        # 2. One-time setup (creates the checkpoint tables in Neon if they don't exist)
        await saver.setup()
        # 3. Compile the graph WITH the checkpointer
        agent_app = workflow.compile(checkpointer=saver)

        print("🚀 Wealth Advisor Agent is Online & Connected to Neon")
        yield  # The FastAPI server runs here

    print("🔌 Shutting down and closing connections...")


app = FastAPI(title="AI Wealth Advisor", lifespan=lifespan)


# --- API Models ---
class ChatInput(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatOutput(BaseModel):
    answer: str
    thread_id: str
    download_url: Optional[str] = None


# --- The Chat Endpoint ---
@app.post("/chat", response_model=ChatOutput)
async def chat_with_agent(
    message: str = Form(...),
    thread_id: str = Form(None),
    file: Optional[UploadFile] = File(None)
):
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    file_path = None

    try:
        # 1. Handle optional file
        if file:
            file_extension = os.path.splitext(file.filename)[1]
            local_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(UPLOAD_DIR, local_filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 🔥 async ingestion (DO NOT block chat)
            result_parsed = await ingest_pdf_to_pgvector(file_path, thread_id)

            parsed_text = "\n\n".join([
                page.markdown for page in result_parsed.markdown.pages if page.markdown
            ])
            input_state = {
                "messages": [
                    (
                        "user",
                        f"""
                   
                    <user_question>
                    {message}
                    </user_question>
                    
                    <document>
                    {parsed_text}
                    </document>

                   """
                    )
                ]
            }
            # 2. Run agent immediately
            result = await agent_app.ainvoke(input_state, config=config)

            final_answer = result["messages"][-1].content

            return ChatOutput(answer=final_answer, thread_id=thread_id,)
        input_state = {
            "messages": [
                (
                    "user",
                    f"""
                    <user_question>
                    {message}
                    </user_question>
                           """
                )
            ]
        }
        result = await agent_app.ainvoke(input_state, config=config)

        final_answer = result["messages"][-1].content

        return ChatOutput(answer=final_answer, thread_id=thread_id, download_url=result.get("download_url"))

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


ingestion_status = {}


@app.post("/ingest")
async def ingest_document(
    payload: ChatInput,
    file: UploadFile = File(...),
    thread_id: str = Form(...),
):
    config = {"configurable": {"thread_id": thread_id}}

    try:

        file_extension = os.path.splitext(file.filename)[1]
        local_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, local_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 🔥 Mark as processing
        ingestion_status[thread_id] = "processing"

        result = await ingest_pdf_to_pgvector(file_path, thread_id)
        parsed_text = "\n\n".join([
            page.markdown for page in result.markdown.pages if page.markdown
        ])
        max_chars = 100000  # adjust depending on model

        if len(parsed_text) > max_chars:
            parsed_text = parsed_text[:max_chars]
        input_state = {
            "messages": [
                (
                    "user",
                    f"""
        User question:
        {payload.message}

        Document:
        {parsed_text}
        """
                )
            ]
        }
        result = await agent_app.ainvoke(input_state, config=config)

        # Extract the last message content from the graph state
        final_answer = result["messages"][-1].content

        return ChatOutput(answer=final_answer, thread_id=thread_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


EXPORT_DIR = "exports"


@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(EXPORT_DIR, filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )