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
import asyncio

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
    # Create an asyncio.Queue to collect streamed events.
    event_queue = asyncio.Queue()

    # Emitter: push events (dictionaries) into the queue.
    async def emitter(event: dict):
        await event_queue.put(event)

    # Launch the streaming pipeline task.
    stream_task = asyncio.create_task(pipeline.run_stream(request, emitter))

    if request.stream:

        async def stream_generator():
            # Send a single opening tag for the thinking block.
            opening_event = {"choices": [{"delta": {"content": "<think>\n"}}]}
            yield f"data: {json.dumps(opening_event)}\n\n"
            thinking_closed = False

            # Process and stream events until done.
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    break

                if event.get("type") in ["message", "replace"]:
                    # If this is the final answer event (from pipeline.run_stream),
                    # close the thinking block first.
                    if event.get("final"):
                        if not thinking_closed:
                            closing_event = {"choices": [{"delta": {"content": "\n</think>"}}]}
                            yield f"data: {json.dumps(closing_event)}\n\n"
                            thinking_closed = True
                        # Send the final answer (e.g. mermaid and final response) separately.
                        chunk = {
                            "choices": [{"delta": {"content": event["data"].get("reasoning_content", "")}}]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        # For intermediate tokens, strip any accidental <think> markers.
                        token = event["data"].get("reasoning_content", "")
                        token = token.replace("<think>\n", "").replace("\n</think>", "")
                        chunk = {"choices": [{"delta": {"content": token}}]}
                        yield f"data: {json.dumps(chunk)}\n\n"

                if event.get("done"):
                    break

            # Send done signal after everything.
            yield "data: [DONE]\n\n"
            await stream_task

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    else:
        # Non-streaming: accumulate all tokens.
        collected = ""
        in_block = False
        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=30)
            except asyncio.TimeoutError:
                break
            if event.get("type") in ["message", "replace"]:
                token = event["data"].get("reasoning_content", "")
                # Only start a <think> block once.
                if not in_block:
                    collected += "<think>\n"
                    in_block = True
                collected += token
                if event.get("block_end", False):
                    collected += "\n</think>"
                    in_block = False
            if event.get("done"):
                # If still in a block, close it.
                if in_block:
                    collected += "\n</think>"
                    in_block = False
                break
        await stream_task

        # Remove any trailing closing </think> so it can be placed correctly
        collected = collected.rstrip()
        if collected.endswith("</think>"):
            collected = collected[: -len("</think>")].rstrip()

        # Build and return the complete JSON response.
        # Now the final response will be wrapped with the intermediate thinking block
        # and the final answer will follow.
        chat_response = ChatCompletionResponse(
            model=request.model,
            choices=[
                ChoiceModel(
                    message=MessageModel(
                        reasoning_content=collected,
                        content=collected,
                    )
                )
            ],
        )
        return JSONResponse(content=chat_response.model_dump())


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
