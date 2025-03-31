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
import json
import os

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from loguru import logger
import uvicorn
import httpx

from utils.classes import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CONTACT_US_MAP,
    MessageModel,
    ChoiceModel,
)
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
            self.buffer = event.get("data", {}).get("reasoning_content", "")
        else:
            self.buffer += event.get("data", {}).get("reasoning_content", "")

    def get_buffer(self) -> str:
        return self.buffer.strip()


# ----------------------------------------------------------------------
# FastAPI App and Endpoints
# ----------------------------------------------------------------------
app = FastAPI(
    title="OpenAI Compatible API with MCTS",
    description="Wraps LLM invocations with Monte Carlo Tree Search refinement",
    version="0.0.91",
    root_path="/v1",
    contact=CONTACT_US_MAP,
)
# Create a router
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
# Defining routers
model_router = APIRouter(prefix="/models", tags=["Model Management"])
chat_router = APIRouter(prefix="/chat", tags=["Chat Completions"])

pipeline = Pipeline(
    openai_api_base_url=OPENAI_API_BASE_URL, openai_api_key=OPENAI_API_KEY
)


@chat_router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    Handles chat completion requests by processing input through a pipeline and
    returning the generated response. Supports both streaming and non-streaming
    modes based on the request.

    ## Args:
    - `request` (`ChatCompletionRequest`): The input request containing model
            details and streaming preference.

    ## Returns:
    - `dict` or `StreamingResponse`: A JSON response with the generated chat
        completion, either as a single response or streamed chunks.
    """

    aggregator = EventAggregator()
    final_text = await pipeline.run(request, aggregator)

    def build_response() -> dict:
        chat_response = ChatCompletionResponse(
            model=request.model,
            choices=[
                ChoiceModel(
                    message=MessageModel(
                        reasoning_content=aggregator.get_buffer(),
                        content=final_text,
                    )
                )
            ],
        )
        return chat_response.model_dump()

    if request.stream:
        # Fake streaming
        async def single_chunk() -> AsyncGenerator[str, None]:
            yield json.dumps(build_response())

        return StreamingResponse(single_chunk(), media_type="application/json")
    else:
        return build_response()


@model_router.get("", response_description="Proxied JSON Response")
async def list_models():
    """
    Asynchronously fetches the list of models from the OpenAI API.
    Sends a `GET` request to the models endpoint using an HTTP client and returns
    the response data as a JSON response. Raises an `HTTPException` if the request fails.

    ## Returns:
    - `JSONResponse`: The response containing the list of models.
    ## Raises:
    - `HTTPException`: If the API request fails with a non-200 status code.
    """

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


app.include_router(model_router)
app.include_router(chat_router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
