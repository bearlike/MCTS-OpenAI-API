#!/usr/bin/env python3
# MCTS Components: Node, Prompts, and Agent
from typing import Awaitable, Callable, List, Optional
from loguru import logger
import random
import math
import re

from utils.llm import LLMClient
from utils.classes import ReasoningEffort


# Default MCTS parameters (serving as a fallback)
# Controls exploration vs exploitation trade-off in UCT formula
# Higher values (>1) favor exploration of less-visited nodes
# Default √2 (≈1.414) is theoretically optimal for many MCTS applications
EXPLORATION_WEIGHT = 1.414

# Number of complete MCTS iterations to perform
# Each iteration involves multiple simulations to build the search tree
# Higher values allow more thorough search but increase computation time
DEFAULT_MAX_ITERATIONS = 2

# Number of simulations to run per iteration
# Each simulation expands the tree and evaluates a new potential response
# Higher values provide more accurate node value estimates
DEFAULT_MAX_SIMULATIONS = 2

# Maximum number of child nodes allowed per parent
# Limits branching factor of the tree to manage computational complexity
# Lower values focus search but might miss potential good responses
DEFAULT_MAX_CHILDREN = 2


class Node:
    """
    Represents a node in the Monte Carlo Tree Search.
    Tracks visit counts, value estimates, and maintains parent-child relationships.
    """

    def __init__(
        self,
        content: str,
        parent: Optional["Node"] = None,
        exploration_weight: float = EXPLORATION_WEIGHT,
        max_children: int = DEFAULT_MAX_CHILDREN,
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
        """Adds a child node and sets its parent reference."""
        child.parent = self
        self.children.append(child)

    def fully_expanded(self) -> bool:
        """Returns True if the node has reached the maximum allowed children."""
        return len(self.children) >= self.max_children

    def uct_value(self) -> float:
        """
        Calculates Upper Confidence Bound for Trees (UCT) value.
        Balances exploration and exploitation.
        """
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + self.exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self) -> "Node":
        """Returns the child node with the highest visit count recursively."""
        if not self.children:
            return self
        return max(self.children, key=lambda child: child.visits).best_child()

    def get_mermaid_lines(self) -> str:
        """
        Generates Mermaid diagram markup representing the tree structure.
        Includes node IDs, visit counts, and content previews.

        Produce a valid Mermaid diagram:
         - A line `graph TD` is first.
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
    """
    Collection of prompt templates for different MCTS operations.
    Includes templates for initial prompts, thoughts generation, updates, and evaluation.
    """

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
    """
    Implements Monte Carlo Tree Search for iterative response refinement using LLMs.
    Manages tree exploration, evaluation, and response generation.
    """

    def __init__(
        self,
        model: str,
        question: str,
        root_content: str,
        llm_client: LLMClient,
        event_emitter: Callable[[dict], Awaitable[None]],
        reasoning_effort: ReasoningEffort = ReasoningEffort.NORMAL,
    ):
        self.model = model
        self.question = question
        self.llm_client = llm_client
        self.event_emitter = event_emitter
        # Configure MCTS parameters based on the desired reasoning effort
        if reasoning_effort == ReasoningEffort.NORMAL:
            self.max_iterations = 2  # minimum 2 iterations
            self.max_simulations = 2
            self.max_children = 2
        elif reasoning_effort == ReasoningEffort.MEDIUM:
            self.max_iterations = 3
            self.max_simulations = 3
            self.max_children = 3
        elif reasoning_effort == ReasoningEffort.HIGH:
            self.max_iterations = 4
            self.max_simulations = 4
            self.max_children = 4
        else:
            # Fallback to normal if unrecognized effort
            self.max_iterations = 2
            self.max_simulations = 2
            self.max_children = 2

        # Initialize the root node with the agent's max_children setting
        self.root = Node(content=root_content, max_children=self.max_children)
        self.reasoning_effort = reasoning_effort
        self.iteration_responses = []  # List to store iteration details

    async def search(self) -> str:
        """
        Executes MCTS algorithm to find optimal response.
        Returns best answer found after specified iterations.
        """
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

        for i in range(1, self.max_iterations + 1):
            await self.emit_status(f"Iteration {i}/{self.max_iterations}")
            iteration_responses = []
            for _ in range(self.max_simulations):
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

        await self.emit_message(
            f"\n\n---\n<details>\n<summary>Best Answer:</summary>\n\n{best_answer}\n\n</details>\n\n</think>\n"
        )
        return best_answer

    async def select(self, node: Node) -> Node:
        """
        Selects a promising node for expansion using UCT selection.
        Returns leaf node for further exploration.
        """
        while node.fully_expanded() and node.children:
            node = max(node.children, key=lambda n: n.uct_value())
        return node

    async def expand(self, node: Node) -> Node:
        """
        Creates a new child node with improved content based on LLM suggestions.
        Returns the newly created child node.
        """
        thought = await self.generate_thought(node.content)
        new_content = await self.update_approach(node.content, thought)
        # Create child node using the configured max_children value
        child = Node(
            content=new_content,
            parent=node,
            exploration_weight=EXPLORATION_WEIGHT,
            max_children=self.max_children,
        )
        node.add_child(child)
        return child

    async def simulate(self, node: Node) -> float:
        """
        Evaluates node's content quality using LLM scoring.
        Returns a numerical score for the response.
        """
        return await self.evaluate_answer(node.content)

    def backpropagate(self, node: Node, score: float):
        """
        Updates visit counts and value estimates up the tree.
        Propagates simulation results to ancestor nodes.
        """
        while node:
            node.visits += 1
            node.value += score
            node = node.parent

    async def generate_completion(self, prompt: str) -> str:
        """
        Gets a streaming completion from the LLM for a given prompt.
        Returns the accumulated response content.
        """
        messages = [{"role": "user", "content": prompt}]
        content = ""
        async for token in self.llm_client.get_streaming_completion(
            messages, self.model
        ):
            content += token
            await self.emit_message(token)
        return content

    async def generate_thought(self, answer: str) -> str:
        """
        Generates an improvement suggestion for the current answer using LLM.
        Returns the critique as a string.
        """
        prompt = MCTSPromptTemplates.thoughts_prompt.format(
            question=self.question, answer=answer
        )
        return await self.generate_completion(prompt)

    async def update_approach(self, answer: str, critique: str) -> str:
        """
        Revises the answer based on the provided critique using LLM.
        Returns the updated answer incorporating the feedback.
        """
        prompt = MCTSPromptTemplates.update_prompt.format(
            question=self.question, answer=answer, critique=critique
        )
        return await self.generate_completion(prompt)

    async def evaluate_answer(self, answer: str) -> float:
        """
        Scores the answer quality using LLM evaluation.
        Returns a numerical score between 1-10.
        """
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
        """
        Sends a tree visualization and iteration details to the UI.
        Updates the progress display with the current search state.
        """
        mermaid = "```mermaid\n" + self.root.get_mermaid_lines() + "\n```"
        iterations = ""
        for itr in self.iteration_responses:
            iterations += f"\nIteration {itr['iteration']}:\n"
            for resp in itr["responses"]:
                iterations += f"- Node `{resp['node_id']}`: Score `{resp['score']}`\n"
                iterations += f"  - **Response**: {resp['content']}\n"
        msg = f"<think>\n\n<details>\n<summary>Expand to View Intermediate Iterations</summary>\n\n{mermaid}\n{iterations}\n\n</details>\n"
        await self.emit_replace(msg)

    async def emit_message(self, message: str):
        await self.event_emitter({"type": "message", "data": {"content": message}})

    async def emit_status(self, message: str):
        await self.event_emitter({"type": "status", "data": {"description": message}})

    async def emit_replace(self, content: str):
        await self.event_emitter({"type": "replace", "data": {"content": content}})
