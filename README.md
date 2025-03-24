# MCTS OpenAI API Wrapper

![Comparison of Response](docs/screenshot_1.png)

Monte Carlo Tree Search (MCTS) is a method that uses extra compute to explore different candidate responses before selecting a final answer. It works by building a tree of options and running multiple iterations. This is similar in concept to inference scaling, but here a model generates several output candidates, reitereates and picks the best one. Every incoming request is wrapped with a MCTS pipeline to iteratively refine language model outputs.

## Overview

This FastAPI server exposes two endpoints:

| Method | Endpoint               | Description                                                                   |
|--------|------------------------|-------------------------------------------------------------------------------|
| POST   | `/v1/chat/completions` | Accepts chat completion requests. The call is wrapped with an MCTS refinement |
| GET    | `/v1/models`           | Proxies a request to the underlying LLM provider’s models endpoint             |

During a chat completion call, the server executes an MCTS pipeline that generates intermediate updates (including a Mermaid diagram and iteration details). All these intermediate responses are aggregated into a single `<details>` block, and the final answer is appended at the end, following a consistent and structured markdown template.

## Getting Started

### Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org) for dependency management

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/bearlike/mcts-openai-api.git
   cd mcts-openai-api
   ```

2. **Copy the Environment File:**

   Copy the example environment file to `.env` and set your `OPENAI_API_KEY`:

   ```bash
   cp .env.example .env
   ```

   Open the `.env` file and update the `OPENAI_API_KEY` (and other settings if needed).

3. **Install Dependencies:**

   Use Poetry to install the required packages:

   ```bash
   poetry install
   ```

4. **Run the Server:**

   Start the FastAPI server with Uvicorn:

   ```bash
   # Visit http://server-ip:8000/docs to view the Swagger API documentation
   uvicorn main:app --reload
   ```

## Testing the Server

You can test the server using `curl` or any HTTP client.

### Example Request

```bash
curl -X 'POST' \
  'http://mcts-server:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": "How many R in STRAWBERRY?"
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.5
}' | jq -r '.choices[0].message.content'
```

This request will return a JSON response with the aggregated intermediate responses wrapped inside a single `<details>` block, followed by the final answer.

---

## Endpoints

### POST /v1/chat/completions

- **Description:**
  Wraps a chat completion request in an MCTS pipeline that refines the answer by generating intermediate updates and a final response.

- **Request Body Parameters:**

    - `model`: string (e.g., `"gpt-4o-mini"`)
    - `messages`: an array of chat messages (with `role` and `content` properties)
    - `max_tokens`: (optional) number
    - `temperature`: (optional) number
    - `stream`: (optional) boolean (if enabled, aggregates intermediate responses with the final answer in one JSON response)

### GET /v1/models

- **Description:**
  Proxies requests to list available models from the underlying LLM provider using the `OPENAI_API_BASE_URL`.

## License

This project is licensed under the MIT License.
