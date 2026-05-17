# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
import shutil
import os
import uuid
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse
from services.multimodal_rag import ingest_pdf_to_pgvector
import json
from typing import Optional

# LangGraph Imports
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from agent.utils.config import NEON_DATABASE_URL
from agent.utils.workflow import workflow  # Import your StateGraph

# --- Global Graph Instance ---
# We define this globally so the API routes can access the compiled graph
agent_app = None


# --- Lifespan: The "Mac-Friendly" Connection Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_app
    # 1. Initialize the Async Checkpointer (Persistent Memory)
    # Using 'autocommit=True' is a 2026 requirement for Neon/Postgres
    async with AsyncPostgresSaver.from_conn_string(NEON_DATABASE_URL) as saver:
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


@app.post("/chat/stream")
async def chat_stream(payload: ChatInput):
    # Check if the agent is actually loaded
    if agent_app is None:
        raise HTTPException(status_code=503, detail="Agent is still initializing...")

    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        # Force a small 'yield' to open the connection immediately
        yield ": connection established\n\n"

        try:
            # Note: Ensure you are using astream_events correctly on the compiled app
            async for event in agent_app.astream_events(
                    {"messages": [("user", payload.message)]},
                    config,
                    version="v2"
            ):
                kind = event["event"]

                # 1. Catching tokens from the Chat Model
                if kind == "on_chat_model_stream":
                    # Sometimes it's event['data']['chunk'].content
                    # Sometimes it's event['data']['chunk'].text
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, 'content'):
                        content = chunk.content
                        if content:
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

                # 2. Catching Tool Starts (Tavily/Neon)
                elif kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool', 'content': event['name']})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
               User question:
               {message}

               Document:
               {parsed_text}
               """
                )
            ]
        }
        # 2. Run agent immediately
        result = await agent_app.ainvoke(input_state, config=config)

        final_answer = result["messages"][-1].content

        return ChatOutput(answer=final_answer, thread_id=thread_id)

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



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
