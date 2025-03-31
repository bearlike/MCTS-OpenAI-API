#!/usr/bin/env python3
# LLM Client (Wrapper for LangChain OpenAI Chat Model)
from typing import Awaitable, Callable
from fastapi import HTTPException
from loguru import logger

from utils.mcts import MCTSPromptTemplates, MCTSAgent
from utils.classes import ChatCompletionRequest
from utils.llm import LLMClient


class Pipeline:
    """Pipeline: Wraps an incoming request into the MCTS process"""

    def __init__(self, *args, **kwargs):
        self.llm_client = LLMClient(*args, **kwargs)

    async def run(
        self,
        request_body: ChatCompletionRequest,
        emitter: Callable[[dict], Awaitable[None]],
    ) -> str:
        model = request_body.model
        if not request_body.messages:
            raise HTTPException(status_code=400, detail="No messages provided.")

        # * Monkey merging latest and previous messages
        system_messages = "\n".join(
            msg.content + "\n"
            for msg in request_body.messages
            if msg.role.lower() in ["system", "developer"]
        )
        rest_messages = [
            msg
            for msg in request_body.messages
            if msg.role.lower() not in ["system", "developer"]
        ]

        latest = rest_messages[-1].content.strip()
        previous = "\n".join(
            f"{msg.role.capitalize()}: {msg.content}" for msg in rest_messages[:-1]
        )

        question = MCTSPromptTemplates.thread_prompt.format(
            question=latest, messages=previous
        )
        initial_prompt = MCTSPromptTemplates.initial_prompt.format(question=question)
        messages = [{"role": "user", "content": initial_prompt}]

        # Check system_messages (not messages) before inserting.
        if system_messages.strip():
            logger.debug(f"Injecting system prompt - {system_messages}")
            messages.insert(0, {"role": "system", "content": system_messages})

        init_reply = await self.llm_client.get_completion(messages, model)
        mcts_agent = MCTSAgent(
            root_content=init_reply,
            llm_client=self.llm_client,
            question=question,
            event_emitter=emitter,
            reasoning_effort=request_body.reasoning_effort,
            model=model,
        )
        final_answer = await mcts_agent.search()
        final_mermaid = (
            "\n```mermaid\n" + mcts_agent.root.get_mermaid_lines() + "\n```\n"
        )
        response_message = f"{final_mermaid}\n## Final Response\n{final_answer}"
        return response_message

    async def run_stream(
        self,
        request: ChatCompletionRequest,
        emitter: Callable[[dict], Awaitable[None]],
    ) -> None:
        model = request.model
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided.")

        # Merge system and conversation messages.
        system_messages = "\n".join(
            msg.content + "\n"
            for msg in request.messages
            if msg.role.lower() in ["system", "developer"]
        )
        rest_messages = [
            msg
            for msg in request.messages
            if msg.role.lower() not in ["system", "developer"]
        ]

        latest = rest_messages[-1].content.strip()
        previous = "\n".join(
            f"{msg.role.capitalize()}: {msg.content}" for msg in rest_messages[:-1]
        )

        question = MCTSPromptTemplates.thread_prompt.format(
            question=latest, messages=previous
        )
        initial_prompt = MCTSPromptTemplates.initial_prompt.format(question=question)
        messages = [{"role": "user", "content": initial_prompt}]

        if system_messages.strip():
            logger.debug(f"Injecting system prompt - {system_messages}")
            messages.insert(0, {"role": "system", "content": system_messages})

        # Get the initial reply from the LLM.
        init_reply = await self.llm_client.get_completion(messages, model)
        mcts_agent = MCTSAgent(
            root_content=init_reply,
            llm_client=self.llm_client,
            question=question,
            event_emitter=emitter,
            reasoning_effort=request.reasoning_effort,
            model=model,
        )

        final_answer = await mcts_agent.search()

        final_mermaid = (
            "\n```mermaid\n" + mcts_agent.root.get_mermaid_lines() + "\n```\n"
        )
        # Build the final response message
        # (note: the final answer is _not_ wrapped in a <think> block).
        final_response = f"{final_mermaid}\n## Final Response\n{final_answer}"

        # Emit the closing tag alone.
        await emitter(
            {
                "type": "message",
                "data": {"reasoning_content": "\n</think>"},
            }
        )

        # Then emit the final response message.
        await emitter(
            {
                "type": "message",
                "data": {
                    "reasoning_content": final_response,
                    "content": final_response,
                },
                "done": True,
                "final": True,
            }
        )
