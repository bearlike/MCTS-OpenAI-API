#!/usr/bin/env python3
# LLM Client (Wrapper for LangChain OpenAI Chat Model)
from typing import Any, List, AsyncGenerator
import asyncio

from langchain.schema import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from utils.handlers import AsyncIteratorCallbackHandler


class LLMClient:
    def __init__(self, openai_api_base_url: str, openai_api_key: str):
        self.base_url = openai_api_base_url
        self.api_key = openai_api_key

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
