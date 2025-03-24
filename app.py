#!/usr/bin/env python3
"""
OpenAI-Compatible FastAPI Server with MCTS wrapping

This FastAPI server exposes two endpoints:
 • POST /v1/chat/completions – for chat completions.
 • GET /v1/models – a simple proxy to the underlying LLM provider models endpoint.

Each Chat Completion call is wrapped in a Monte Carlo Tree Search (MCTS)
refinement process. Intermediate updates (Mermaid diagram and iteration details)
are accumulated in a single <details> block and then returned together with the final answer.
"""

from typing import AsyncGenerator
import time
import json
import os

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from loguru import logger
import httpx

from utils.classes import ChatCompletionRequest
from utils.llm.pipeline import Pipeline

load_dotenv()

# Global Configuration (from ENV)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)  # Must be set
OPENAI_API_BASE_URL = os.environ.get(
    "OPENAI_API_BASE_URL", "http://lite-llm-proxy:4000/v1"
)
if OPENAI_API_KEY is None:
    _msg = "OPENAI_API_KEY is not set. Please set it in the environment."
    logger.error(_msg)
    raise ValueError(_msg)

logger.info(f"Using OpenAI API Base URL: {OPENAI_API_BASE_URL}")


# ----------------------------------------------------------------------
# Event Aggregator: For final message assembly
# ----------------------------------------------------------------------
class EventAggregator:
    def __init__(self):
        self.buffer = ""

    async def __call__(self, event: dict):
        if event.get("type") == "replace":
            self.buffer = event.get("data", {}).get("content", "")
        else:
            self.buffer += event.get("data", {}).get("content", "")

    def get_buffer(self) -> str:
        return self.buffer


# ----------------------------------------------------------------------
# FastAPI App and Endpoints
# ----------------------------------------------------------------------
app = FastAPI(
    title="OpenAI Compatible API with MCTS",
    description="Wraps LLM invocations with Monte Carlo Tree Search refinement",
    version="0.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
pipeline = Pipeline(
    openai_api_base_url=OPENAI_API_BASE_URL, openai_api_key=OPENAI_API_KEY
)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        aggregator = EventAggregator()
        final_text = await pipeline.run(request, aggregator)
        full_message = aggregator.get_buffer() + "\n" + final_text
        final_response = {
            "id": "mcts_response",
            "object": "chat.completion",
            "created": time.time(),
            "model": request.model,
            "choices": [{"message": {"role": "assistant", "content": full_message}}],
        }

        # Return a single JSON chunk with mimetype application/json
        async def single_chunk() -> AsyncGenerator[str, None]:
            yield json.dumps(final_response)

        return StreamingResponse(single_chunk(), media_type="application/json")
    else:
        aggregator = EventAggregator()
        final_text = await pipeline.run(request, aggregator)
        full_message = aggregator.get_buffer() + "\n" + final_text
        return {
            "id": "mcts_response",
            "object": "chat.completion",
            "created": time.time(),
            "model": request.model,
            "choices": [{"message": {"role": "assistant", "content": full_message}}],
        }


@app.get("/v1/models")
async def list_models():
    url = f"{OPENAI_API_BASE_URL}/models"
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            url, headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
        )
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code, detail="Failed to proxy models endpoint."
            )
        data = resp.json()
    return JSONResponse(content=data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
