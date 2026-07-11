#!/usr/bin/env python3

import asyncio
import json
import os
import sys
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


LLAMA_URL = os.environ.get(
    "LLAMA_URL",
    "http://127.0.0.1:8080/v1/chat/completions",
)

WORKSPACE = os.environ.get(
    "MCP_WORKSPACE",
    "/home/abagx/mcp-workspace",
)

MAX_AGENT_STEPS = 12


def mcp_tools_to_openai(
    mcp_tools: list[Any],
) -> list[dict[str, Any]]:
    """Convert MCP tools to OpenAI-compatible function definitions."""

    converted: list[dict[str, Any]] = []

    for tool in mcp_tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
        )

    return converted


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Convert model-generated tool arguments to a dictionary."""

    if arguments is None:
        return {}

    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        arguments = arguments.strip()

        if not arguments:
            return {}

        decoded = json.loads(arguments)

        if not isinstance(decoded, dict):
            raise ValueError(
                "Tool arguments must decode to a JSON object"
            )

        return decoded

    raise TypeError(
        "Unsupported tool argument type: "
        f"{type(arguments).__name__}"
    )


def strip_tool_wrappers(content: str) -> str:
    """
    Remove wrappers Qwen may place around textual tool calls.

    Supported examples:

    ```json
    {"name": "...", "arguments": {...}}
    ```

    <tools>
    {"name": "...", "arguments": {...}}
    </tools>
    """

    content = content.strip()

    if not content:
        return content

    # Remove Markdown code fences.
    if content.startswith("```"):
        lines = content.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

    # Remove Qwen-style XML wrapper.
    if content.startswith("<tools>") and content.endswith("</tools>"):
        content = content[
            len("<tools>") : -len("</tools>")
        ].strip()

    # Some templates use singular <tool_call>.
    if (
        content.startswith("<tool_call>")
        and content.endswith("</tool_call>")
    ):
        content = content[
            len("<tool_call>") : -len("</tool_call>")
        ].strip()

    return content


def parse_text_tool_call(
    content: str,
) -> list[dict[str, Any]]:
    """
    Parse a tool call emitted by the model as ordinary text.

    Expected JSON form:

    {
      "name": "list_directory",
      "arguments": {
        "path": "/home/abagx/mcp-workspace"
      }
    }
    """

    content = strip_tool_wrappers(content)

    if not content:
        return []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []

    # Support one tool object.
    if isinstance(parsed, dict):
        parsed_calls = [parsed]

    # Also support a JSON list of tool objects.
    elif isinstance(parsed, list):
        parsed_calls = parsed

    else:
        return []

    tool_calls: list[dict[str, Any]] = []

    for index, parsed_call in enumerate(parsed_calls):
        if not isinstance(parsed_call, dict):
            continue

        name = parsed_call.get("name")
        arguments = parsed_call.get("arguments", {})

        if not isinstance(name, str) or not name:
            continue

        if not isinstance(arguments, dict):
            continue

        tool_calls.append(
            {
                "id": f"text-tool-call-{index + 1}",
                "type": "function",
                "from_text": True,
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )

    return tool_calls


def mcp_result_to_text(result: Any) -> str:
    """Convert an MCP result into text suitable for the model."""

    output: list[str] = []

    for item in result.content:
        if isinstance(item, types.TextContent):
            output.append(item.text)
        else:
            try:
                output.append(item.model_dump_json())
            except AttributeError:
                output.append(str(item))

    text = "\n".join(output)

    if result.isError:
        return "MCP tool error:\n" + text

    return text


async def call_llama(
    client: httpx.AsyncClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call llama-server through its OpenAI-compatible endpoint."""

    payload = {
        "model": "qwen2.5-coder",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.1,
        "max_tokens": 3000,
        "stream": False,
    }

    response = await client.post(
        LLAMA_URL,
        json=payload,
        timeout=300.0,
    )

    response.raise_for_status()
    body = response.json()

    try:
        return body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "Unexpected llama-server response:\n"
            + json.dumps(body, indent=2)
        ) from error


async def execute_tool(
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """Execute one MCP tool call."""

    try:
        result = await session.call_tool(
            tool_name,
            arguments=arguments,
        )

        return mcp_result_to_text(result)

    except Exception as error:
        return (
            "Tool execution failed: "
            f"{type(error).__name__}: {error}"
        )


def make_text_tool_result_message(
    tool_name: str,
    arguments: dict[str, Any],
    result_text: str,
) -> dict[str, str]:
    """
    Build a message for tool calls that Qwen emitted as ordinary text.

    Because the original request was not a native OpenAI tool_call,
    sending role='tool' may confuse the model. A clearly labelled user
    message is more reliable for this fallback path.
    """

    return {
        "role": "user",
        "content": (
            f"The filesystem tool `{tool_name}` has completed.\n\n"
            "Tool arguments:\n"
            f"{json.dumps(arguments, ensure_ascii=False, indent=2)}\n\n"
            "Actual tool result:\n"
            f"{result_text}\n\n"
            "Now answer the original user request using the actual "
            "tool result above. Give the concrete answer. Do not merely "
            "acknowledge receipt of a tool response. Do not repeat the "
            "tool request unless another filesystem operation is truly "
            "necessary."
        ),
    }


async def run_agent(
    session: ClientSession,
    openai_tools: list[dict[str, Any]],
) -> None:
    """Run the interactive user/agent/tool loop."""

    system_prompt = (
        "You are a local coding assistant with filesystem tools. "
        f"The only allowed filesystem root is `{WORKSPACE}`. "
        "Always use that exact absolute path or a path below it. "
        "Never use `/` as the filesystem path. "
        "Use a supplied filesystem tool whenever the user asks about "
        "files or directories. "
        "Prefer reading before editing. "
        "Never create, edit, move, or delete files unless the user "
        "explicitly asks you to do so. "
        "When a tool result is provided, use its actual contents to "
        "answer the original request. "
        "Do not merely say that a tool response was provided. "
        "Do not print a tool call as the final answer."
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    async with httpx.AsyncClient() as http_client:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                print("\nExiting.")
                return
            except KeyboardInterrupt:
                print("\nExiting.")
                return

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                print("Exiting.")
                return

            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            for step in range(MAX_AGENT_STEPS):
                try:
                    assistant = await call_llama(
                        http_client,
                        messages,
                        openai_tools,
                    )
                except httpx.HTTPError as error:
                    print(
                        "\nFailed to contact llama-server: "
                        f"{type(error).__name__}: {error}"
                    )
                    break
                except Exception as error:
                    print(
                        "\nModel request failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    break

                native_tool_calls = (
                    assistant.get("tool_calls") or []
                )

                text_tool_calls: list[dict[str, Any]] = []

                if not native_tool_calls:
                    text_tool_calls = parse_text_tool_call(
                        assistant.get("content") or ""
                    )

                tool_calls = (
                    native_tool_calls or text_tool_calls
                )

                assistant_history: dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant.get("content") or "",
                }

                # Only native tool calls belong in the OpenAI
                # tool_calls history field.
                if native_tool_calls:
                    assistant_history["tool_calls"] = (
                        native_tool_calls
                    )

                messages.append(assistant_history)

                if not tool_calls:
                    answer = (
                        assistant.get("content")
                        or "(empty response)"
                    )

                    print(f"\nAssistant: {answer}")
                    break

                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    tool_name = function.get("name")
                    call_id = tool_call.get("id")

                    if not isinstance(tool_name, str) or not tool_name:
                        print(
                            "\nThe model returned a tool call "
                            "without a valid name."
                        )
                        continue

                    try:
                        arguments = parse_tool_arguments(
                            function.get("arguments")
                        )
                    except Exception as error:
                        print(
                            "\nInvalid tool arguments for "
                            f"{tool_name}: {error}"
                        )
                        continue

                    print(
                        f"\n[tool {step + 1}] "
                        f"{tool_name}"
                        f"({json.dumps(arguments, ensure_ascii=False)})"
                    )

                    result_text = await execute_tool(
                        session,
                        tool_name,
                        arguments,
                    )

                    if tool_call.get("from_text"):
                        messages.append(
                            make_text_tool_result_message(
                                tool_name,
                                arguments,
                                result_text,
                            )
                        )
                    else:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": result_text,
                            }
                        )

            else:
                print(
                    "\nStopped after "
                    f"{MAX_AGENT_STEPS} tool-call steps."
                )


async def main() -> None:
    """Start the Filesystem MCP server and interactive bridge."""

    if not os.path.isdir(WORKSPACE):
        print(
            f"Workspace does not exist: {WORKSPACE}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    server = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            WORKSPACE,
        ],
    )

    print(f"Allowed workspace: {WORKSPACE}")
    print(f"llama-server: {LLAMA_URL}")

    try:
        async with stdio_client(
            server
        ) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:
                await session.initialize()

                available = await session.list_tools()

                print("Filesystem MCP tools:")

                for tool in available.tools:
                    print(f"  - {tool.name}")

                openai_tools = mcp_tools_to_openai(
                    available.tools
                )

                await run_agent(
                    session,
                    openai_tools,
                )

    except asyncio.CancelledError:
        return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")