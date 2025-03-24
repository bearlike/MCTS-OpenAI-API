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

from typing import Any, Awaitable, Callable, List, Optional, AsyncGenerator
import asyncio
import random
import math
import time
import json
import re
import os

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger
import httpx

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ----------------------------------------------------------------------
# Global Configuration (from ENV)
# ----------------------------------------------------------------------
OPENAI_API_BASE_URL = os.environ.get(
    "OPENAI_API_BASE_URL", "http://lite-llm-proxy:4000/v1"
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-XXX")  # Must be set

logger.info(f"Using OpenAI API Base URL: {OPENAI_API_BASE_URL}")

# Default MCTS parameters
EXPLORATION_WEIGHT = 1.414
MAX_ITERATIONS = 2
MAX_SIMULATIONS = 2
MAX_CHILDREN = 2


# ----------------------------------------------------------------------
# Pydantic Models for Chat Completion API
# ----------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str  # e.g.: "gpt-4o-mini"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


# ----------------------------------------------------------------------
# Async Iterator Callback for Streaming Tokens (if needed)
# ----------------------------------------------------------------------
class AsyncIteratorCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()
        self.done = False

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.queue.put(token)

    async def on_llm_end(self, response: AIMessage, **kwargs: Any) -> None:
        self.done = True
        await self.queue.put(None)

    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        self.done = True
        await self.queue.put(None)

    async def __aiter__(self) -> AsyncGenerator[str, None]:
        while not self.done:
            token = await self.queue.get()
            if token is None:
                break
            yield token


# ----------------------------------------------------------------------
# LLM Client (Wrapper for LangChain OpenAI Chat Model)
# ----------------------------------------------------------------------
class LLMClient:
    def __init__(self):
        self.base_url = OPENAI_API_BASE_URL
        self.api_key = OPENAI_API_KEY

    async def create_chat_completion(
        self, messages: List[dict], model: str, stream: bool = False
    ) -> Any:
        lc_messages = []
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))
        if stream:
            handler = AsyncIteratorCallbackHandler()
            oai_model = ChatOpenAI(
                extra_body={"cache": {"no-cache": True}},
                base_url=self.base_url,
                api_key=self.api_key,
                streaming=True,
                model=model,
                cache=False,
                callbacks=[handler],
            )
            asyncio.create_task(oai_model.agenerate([lc_messages]))
            return handler
        else:
            oai_model = ChatOpenAI(
                extra_body={"cache": {"no-cache": True}},
                base_url=self.base_url,
                api_key=self.api_key,
                streaming=False,
                model=model,
                cache=False,
            )
            response = await oai_model.agenerate([lc_messages])
            return response.generations[0][0].message.content

    async def get_streaming_completion(
        self, messages: List[dict], model: str
    ) -> AsyncGenerator[str, None]:
        response = await self.create_chat_completion(messages, model, stream=True)
        async for token in response:
            yield token

    async def get_completion(self, messages: List[dict], model: str) -> str:
        content = await self.create_chat_completion(messages, model, stream=False)
        return content


# ----------------------------------------------------------------------
# MCTS Components: Node, Prompts, and Agent
# ----------------------------------------------------------------------
class Node:
    def __init__(
        self,
        content: str,
        parent: Optional["Node"] = None,
        exploration_weight: float = EXPLORATION_WEIGHT,
        max_children: int = MAX_CHILDREN,
    ):
        self.id = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=4))
        self.content = content
        self.parent = parent
        self.exploration_weight = exploration_weight
        self.max_children = max_children
        self.children: List["Node"] = []
        self.visits = 0
        self.value = 0.0

    def add_child(self, child: "Node"):
        child.parent = self
        self.children.append(child)

    def fully_expanded(self) -> bool:
        return len(self.children) >= self.max_children

    def uct_value(self) -> float:
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + self.exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self) -> "Node":
        if not self.children:
            return self
        return max(self.children, key=lambda child: child.visits).best_child()

    def get_mermaid_lines(self) -> str:
        """
        Produce a valid Mermaid diagram:
         - A line "graph TD" is first.
         - Then all node definitions are inserted, one per line (properly quoted).
         - Then an empty line followed by connection definitions, one per line.
        """
        definitions = {}
        connections = []

        def dfs(node: "Node"):
            preview = node.content.replace('"', "'").replace("\n", " ")[:25]
            definitions[node.id] = f'{node.id}["{node.id}: ({node.visits}) {preview}"]'
            for child in node.children:
                connections.append(f"{node.id} --> {child.id}")
                dfs(child)

        dfs(self)
        lines = ["graph TD"]
        for def_line in definitions.values():
            lines.append("    " + def_line)
        lines.append("")  # blank line
        for con_line in connections:
            lines.append("    " + con_line)
        return "\n".join(lines)


class MCTSPromptTemplates:
    thread_prompt = """
## Latest Question
{question}

## Previous Messages
{messages}
    """

    initial_prompt = """
<instruction>
Provide a clear, accurate, and complete answer to the question below.
</instruction>
<question>
{question}
</question>
    """

    thoughts_prompt = """
<instruction>
In one sentence, provide a suggestion to improve the answer.
</instruction>
<question>
{question}
</question>
<draft>
{answer}
</draft>
    """

    update_prompt = """
<instruction>
Revise the answer below addressing the critique.
Return only the updated answer.
</instruction>
<question>
{question}
</question>
<draft>
{answer}
</draft>
<critique>
{critique}
</critique>
    """

    eval_answer_prompt = """
<instruction>
Score how well the answer responds to the question on a scale of 1 to 10.
Return a single number.
</instruction>
<question>
{question}
</question>
<answer>
{answer}
</answer>
    """


class MCTSAgent:
    def __init__(
        self,
        root_content: str,
        llm_client: LLMClient,
        question: str,
        event_emitter: Callable[[dict], Awaitable[None]],
        model: str,
    ):
        self.root = Node(content=root_content)
        self.question = question
        self.llm_client = llm_client
        self.event_emitter = event_emitter
        self.model = model
        self.iteration_responses = []  # List to store iteration details

    async def search(self) -> str:
        best_answer = None
        best_score = float("-inf")
        processed_ids = set()

        # Evaluate the root node
        root_score = await self.evaluate_answer(self.root.content)
        self.root.visits += 1
        self.root.value += root_score
        processed_ids.add(self.root.id)
        self.iteration_responses.append(
            {
                "iteration": 0,
                "responses": [
                    {
                        "node_id": self.root.id,
                        "content": self.root.content,
                        "score": root_score,
                    }
                ],
            }
        )
        await self.emit_iteration_update()

        for i in range(1, MAX_ITERATIONS + 1):
            await self.emit_status(f"Iteration {i}/{MAX_ITERATIONS}")
            iteration_responses = []
            for _ in range(MAX_SIMULATIONS):
                leaf = await self.select(self.root)
                if not leaf.fully_expanded():
                    child = await self.expand(leaf)
                    if child.id not in processed_ids:
                        score = await self.simulate(child)
                        self.backpropagate(child, score)
                        iteration_responses.append(
                            {
                                "node_id": child.id,
                                "content": child.content,
                                "score": score,
                            }
                        )
                        processed_ids.add(child.id)
                else:
                    if leaf.id not in processed_ids and leaf.id != self.root.id:
                        score = await self.simulate(leaf)
                        self.backpropagate(leaf, score)
                        iteration_responses.append(
                            {
                                "node_id": leaf.id,
                                "content": leaf.content,
                                "score": score,
                            }
                        )
                        processed_ids.add(leaf.id)
            if iteration_responses:
                self.iteration_responses.append(
                    {
                        "iteration": i,
                        "responses": iteration_responses,
                    }
                )
            await self.emit_iteration_update()
            current_node = self.root.best_child()
            current_score = (
                (current_node.value / current_node.visits)
                if current_node.visits > 0
                else 0
            )
            if current_score > best_score:
                best_score = current_score
                best_answer = current_node.content

        await self.emit_message(f"\n\n---\n## Best Answer:\n{best_answer}")
        return best_answer

    async def select(self, node: Node) -> Node:
        while node.fully_expanded() and node.children:
            node = max(node.children, key=lambda n: n.uct_value())
        return node

    async def expand(self, node: Node) -> Node:
        thought = await self.generate_thought(node.content)
        new_content = await self.update_approach(node.content, thought)
        child = Node(
            content=new_content,
            parent=node,
            exploration_weight=EXPLORATION_WEIGHT,
            max_children=MAX_CHILDREN,
        )
        node.add_child(child)
        return child

    async def simulate(self, node: Node) -> float:
        return await self.evaluate_answer(node.content)

    def backpropagate(self, node: Node, score: float):
        while node:
            node.visits += 1
            node.value += score
            node = node.parent

    async def generate_completion(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        content = ""
        async for token in self.llm_client.get_streaming_completion(
            messages, self.model
        ):
            content += token
            await self.emit_message(token)
        return content

    async def generate_thought(self, answer: str) -> str:
        prompt = MCTSPromptTemplates.thoughts_prompt.format(
            question=self.question, answer=answer
        )
        return await self.generate_completion(prompt)

    async def update_approach(self, answer: str, critique: str) -> str:
        prompt = MCTSPromptTemplates.update_prompt.format(
            question=self.question, answer=answer, critique=critique
        )
        return await self.generate_completion(prompt)

    async def evaluate_answer(self, answer: str) -> float:
        prompt = MCTSPromptTemplates.eval_answer_prompt.format(
            question=self.question, answer=answer
        )
        result = await self.generate_completion(prompt)
        try:
            score = int(re.search(r"\d+", result).group())
            return score
        except Exception as e:
            logger.error("Score parsing error: {} from '{}'", e, result)
            return 0

    async def emit_iteration_update(self):
        mermaid = "```mermaid\n" + self.root.get_mermaid_lines() + "\n```"
        iterations = ""
        for itr in self.iteration_responses:
            iterations += f"\nIteration {itr['iteration']}:\n"
            for resp in itr["responses"]:
                iterations += f"- Node `{resp['node_id']}`: Score `{resp['score']}`\n"
                iterations += f"  - **Response**: {resp['content']}\n"
        msg = f"## Intermediate Responses\n<details>\n<summary>Expand to View Intermediate Iterations</summary>\n\n{mermaid}\n{iterations}\n\n</details>\n"
        await self.emit_replace(msg)

    async def emit_message(self, message: str):
        await self.event_emitter({"type": "message", "data": {"content": message}})

    async def emit_status(self, message: str):
        await self.event_emitter({"type": "status", "data": {"description": message}})

    async def emit_replace(self, content: str):
        await self.event_emitter({"type": "replace", "data": {"content": content}})


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
# Pipeline: Wraps an incoming request into the MCTS process
# ----------------------------------------------------------------------
class Pipeline:
    def __init__(self):
        self.llm_client = LLMClient()

    async def run(
        self,
        request_body: ChatCompletionRequest,
        emitter: Callable[[dict], Awaitable[None]],
    ) -> str:
        model = request_body.model
        if not request_body.messages:
            raise HTTPException(status_code=400, detail="No messages provided.")
        latest = request_body.messages[-1].content.strip()
        previous = "\n".join(
            f"{msg.role.capitalize()}: {msg.content}"
            for msg in request_body.messages[:-1]
        )
        question = MCTSPromptTemplates.thread_prompt.format(
            question=latest, messages=previous
        )
        initial_prompt = MCTSPromptTemplates.initial_prompt.format(question=question)
        init_reply = await self.llm_client.get_completion(
            [{"role": "user", "content": initial_prompt}], model
        )
        mcts_agent = MCTSAgent(
            root_content=init_reply,
            llm_client=self.llm_client,
            question=question,
            event_emitter=emitter,
            model=model,
        )
        final_answer = await mcts_agent.search()
        final_mermaid = (
            "\n```mermaid\n" + mcts_agent.root.get_mermaid_lines() + "\n```\n"
        )
        response_message = f"{final_mermaid}\n## Final Response\n{final_answer}"
        return response_message


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
pipeline = Pipeline()


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
