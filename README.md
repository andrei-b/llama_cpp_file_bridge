# file_agent

A local Python bridge between a llama-server OpenAI-compatible endpoint and the MCP filesystem server.

## What it does

- Starts `@modelcontextprotocol/server-filesystem` via `npx`
- Converts MCP tools to OpenAI tool definitions
- Sends user prompts to llama-server
- Executes model-requested filesystem tool calls and returns results

## Requirements

- Python 3.10+
- Node.js + `npx`
- A running llama-server endpoint compatible with OpenAI Chat Completions

Python dependencies are listed in `requirements.txt`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Optional environment variables:

- `LLAMA_URL` (default: `http://127.0.0.1:8080/v1/chat/completions`)
- `MCP_WORKSPACE` (default in code: `/home/abagx/mcp-workspace`)

Example:

```bash
export LLAMA_URL="http://127.0.0.1:8080/v1/chat/completions"
export MCP_WORKSPACE="/home/andrei/CLionProjects/file_agent"
```

## Run

```bash
python3 main.py
```

Type your request at the prompt. Use `exit` or `quit` to stop.

