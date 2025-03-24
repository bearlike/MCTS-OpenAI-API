#!/usr/bin/env python3
# Async Iterator Callback for Streaming Tokens (when needed)
import asyncio
from typing import Any, AsyncGenerator
from langchain.schema import AIMessage
from langchain.callbacks.base import AsyncCallbackHandler


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
