#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


LLAMA_URL = os.environ.get(
    "LLAMA_URL",
    "http://localhost-0:8080/v1/chat/completions",
)

WORKSPACE = str(
    Path(
        os.environ.get(
            "MCP_WORKSPACE",
            "/home/user/project",
        )
    ).resolve()
)

MODEL_NAME = os.environ.get(
    "LLAMA_MODEL",
    "qwen2.5-coder",
)

try:
    TEMPERATURE = float(
        os.environ.get("LLAMA_TEMPERATURE", "0.1")
    )
except ValueError:
    TEMPERATURE = 0.1

try:
    MAX_TOKENS = int(
        os.environ.get("LLAMA_MAX_TOKENS", "3000")
    )
except ValueError:
    MAX_TOKENS = 3000

MAX_AGENT_STEPS = 12
MAX_FILE_SIZE = 2 * 1024 * 1024

EXCLUDED_DIRECTORIES = {
    ".git",
    ".idea",
    ".vscode",
    ".cache",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "build",
    "cmake-build-debug",
    "cmake-build-release",
    "dist",
    "target",
    "out",
    "vendor",
}

SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".py",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".cs",
    ".kt",
    ".kts",
    ".swift",
    ".sh",
}

COMMON_ENTRY_FILENAMES = {
    "main.c",
    "main.cc",
    "main.cpp",
    "main.cxx",
    "main.py",
    "__main__.py",
    "app.py",
    "server.py",
    "manage.py",
    "index.js",
    "index.ts",
    "app.js",
    "app.ts",
    "server.js",
    "server.ts",
    "main.js",
    "main.ts",
    "main.go",
    "main.rs",
    "main.java",
    "program.cs",
}


def is_path_allowed(path: str) -> bool:
    """Return True when path is inside the configured workspace."""

    resolved = os.path.abspath(path)

    try:
        return os.path.commonpath([WORKSPACE, resolved]) == WORKSPACE
    except ValueError:
        return False


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


def custom_tools() -> list[dict[str, Any]]:
    """Tools implemented directly by this bridge."""

    return [
        {
            "type": "function",
            "function": {
                "name": "find_main_program",
                "description": (
                    "Find likely program entry-point files in a software "
                    "project. Supports C, C++, Python, JavaScript, TypeScript, "
                    "Java, Go, Rust, C#, Kotlin and shell projects. It searches "
                    "for main functions, Python __main__ blocks and common "
                    "entry-point filenames."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Absolute project directory inside the allowed "
                                "workspace."
                            ),
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 15,
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}

    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        text = arguments.strip()

        if not text:
            return {}

        decoded = json.loads(text)

        if not isinstance(decoded, dict):
            raise ValueError(
                "Tool arguments must decode to a JSON object"
            )

        return decoded

    raise TypeError(
        f"Unsupported tool argument type: {type(arguments).__name__}"
    )


def extract_json_objects(text: str) -> list[Any]:
    """
    Extract complete JSON objects or arrays from surrounding model text.

    This works with output such as:

        Explanation...
        <tools>
        {"name": "...", "arguments": {...}}
        </tools>

    It also tolerates a missing closing </tools> tag.
    """

    decoder = json.JSONDecoder()
    results: list[Any] = []

    for index, character in enumerate(text):
        if character not in "[{":
            continue

        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue

        results.append(parsed)

    return results


def normalize_text_tool_object(
    value: Any,
    index: int,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    # Qwen sometimes emits:
    # {"name": "...", "arguments": {...}}
    if isinstance(value.get("name"), str):
        name = value["name"]
        arguments = value.get("arguments", {})

    # Some models emit:
    # {"function": {"name": "...", "arguments": {...}}}
    elif isinstance(value.get("function"), dict):
        function = value["function"]
        name = function.get("name")
        arguments = function.get("arguments", {})

    else:
        return None

    if not isinstance(name, str) or not name:
        return None

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None

    if not isinstance(arguments, dict):
        return None

    return {
        "id": f"text-tool-call-{index}",
        "type": "function",
        "from_text": True,
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def parse_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """
    Parse tool calls printed as plain text instead of native tool_calls.

    Handles:
      - raw JSON
      - Markdown JSON blocks
      - <tools>...</tools>
      - explanatory prose before JSON
      - a missing closing </tools> tag
      - one object or a list of objects
    """

    values = extract_json_objects(content)
    calls: list[dict[str, Any]] = []

    for value in values:
        candidates = value if isinstance(value, list) else [value]

        for candidate in candidates:
            normalized = normalize_text_tool_object(
                candidate,
                len(calls) + 1,
            )

            if normalized is not None:
                calls.append(normalized)

    # Avoid duplicate objects extracted from nested JSON positions.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()

    for call in calls:
        key = json.dumps(
            call["function"],
            sort_keys=True,
            ensure_ascii=False,
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(call)

    return unique


def mcp_result_to_text(result: Any) -> str:
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


def read_small_text_file(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return None

        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None


def entry_point_score(
    path: Path,
    content: str,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    filename = path.name.lower()
    suffix = path.suffix.lower()

    if filename in COMMON_ENTRY_FILENAMES:
        score += 35
        reasons.append("common entry-point filename")

    patterns: list[tuple[str, int, str]] = []

    if suffix in {".c", ".cc", ".cpp", ".cxx"}:
        patterns.extend(
            [
                (
                    r"\b(?:int|auto)\s+main\s*\(",
                    100,
                    "contains a C/C++ main() function",
                ),
                (
                    r"\bwmain\s*\(",
                    95,
                    "contains a Windows wmain() function",
                ),
            ]
        )

    elif suffix == ".py":
        patterns.extend(
            [
                (
                    r"""if\s+__name__\s*==\s*["']__main__["']\s*:""",
                    100,
                    "contains a Python __main__ guard",
                ),
                (
                    r"\bdef\s+main\s*\(",
                    40,
                    "defines a Python main() function",
                ),
            ]
        )

    elif suffix in {".js", ".mjs", ".cjs", ".ts", ".tsx"}:
        patterns.extend(
            [
                (
                    r"\bcreateServer\s*\(",
                    35,
                    "creates an HTTP/server entry point",
                ),
                (
                    r"\bapp\.listen\s*\(",
                    70,
                    "starts an application server",
                ),
                (
                    r"\bnew\s+Command\s*\(",
                    25,
                    "appears to initialize a CLI",
                ),
                (
                    r"\bfunction\s+main\s*\(",
                    50,
                    "defines a JavaScript main() function",
                ),
            ]
        )

    elif suffix == ".java":
        patterns.append(
            (
                r"public\s+static\s+void\s+main\s*\(",
                100,
                "contains a Java main() method",
            )
        )

    elif suffix == ".go":
        patterns.extend(
            [
                (
                    r"(?m)^\s*package\s+main\s*$",
                    50,
                    "belongs to Go package main",
                ),
                (
                    r"\bfunc\s+main\s*\(",
                    100,
                    "contains a Go main() function",
                ),
            ]
        )

    elif suffix == ".rs":
        patterns.append(
            (
                r"\bfn\s+main\s*\(",
                100,
                "contains a Rust main() function",
            )
        )

    elif suffix == ".cs":
        patterns.extend(
            [
                (
                    r"\bstatic\s+(?:async\s+)?(?:void|int|Task)\s+Main\s*\(",
                    100,
                    "contains a C# Main() method",
                ),
                (
                    r"WebApplication\.CreateBuilder\s*\(",
                    70,
                    "contains an ASP.NET application entry point",
                ),
            ]
        )

    elif suffix in {".kt", ".kts"}:
        patterns.append(
            (
                r"\bfun\s+main\s*\(",
                100,
                "contains a Kotlin main() function",
            )
        )

    elif suffix == ".swift":
        patterns.extend(
            [
                (
                    r"@main\b",
                    100,
                    "contains a Swift @main declaration",
                ),
                (
                    r"\bfunc\s+main\s*\(",
                    70,
                    "contains a Swift main() function",
                ),
            ]
        )

    elif suffix == ".sh":
        if content.startswith("#!"):
            score += 15
            reasons.append("executable shell script")

    for pattern, points, description in patterns:
        if re.search(pattern, content):
            score += points
            reasons.append(description)

    return score, reasons


def inspect_build_metadata(root: Path) -> list[dict[str, Any]]:
    """
    Find entry-point hints in common build metadata.

    This does not replace source inspection, but helps identify CMake and
    package-based application targets.
    """

    hints: list[dict[str, Any]] = []

    for name in (
        "CMakeLists.txt",
        "package.json",
        "pyproject.toml",
        "Cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
    ):
        for path in root.rglob(name):
            if any(part in EXCLUDED_DIRECTORIES for part in path.parts):
                continue

            content = read_small_text_file(path)

            if content is None:
                continue

            details: list[str] = []

            if name == "CMakeLists.txt":
                matches = re.findall(
                    r"add_executable\s*\(\s*([^\s\)]+)",
                    content,
                    flags=re.IGNORECASE,
                )

                if matches:
                    details.append(
                        "CMake executable targets: "
                        + ", ".join(matches[:10])
                    )

            elif name == "package.json":
                try:
                    package = json.loads(content)

                    if isinstance(package.get("main"), str):
                        details.append(
                            f"package main: {package['main']}"
                        )

                    scripts = package.get("scripts", {})

                    if isinstance(scripts, dict):
                        for script_name in ("start", "serve", "dev"):
                            if script_name in scripts:
                                details.append(
                                    f"{script_name} script: "
                                    f"{scripts[script_name]}"
                                )
                except json.JSONDecodeError:
                    pass

            elif name == "pyproject.toml":
                if "[project.scripts]" in content:
                    details.append(
                        "contains Python project scripts"
                    )

            elif name == "Cargo.toml":
                if "[[bin]]" in content:
                    details.append("contains Rust binary targets")

            elif name == "go.mod":
                details.append("Go module root")

            if details:
                hints.append(
                    {
                        "file": str(path),
                        "details": details,
                    }
                )

    return hints


def find_main_program(arguments: dict[str, Any]) -> str:
    requested_path = arguments.get("path", WORKSPACE)
    max_results = arguments.get("max_results", 15)

    if not isinstance(requested_path, str):
        return json.dumps(
            {"error": "path must be a string"},
            indent=2,
        )

    if not isinstance(max_results, int):
        max_results = 15

    max_results = max(1, min(max_results, 50))
    root = Path(requested_path).resolve()

    if not is_path_allowed(str(root)):
        return json.dumps(
            {
                "error": "Path is outside the allowed workspace",
                "requested_path": str(root),
                "allowed_workspace": WORKSPACE,
            },
            indent=2,
        )

    if not root.exists():
        return json.dumps(
            {
                "error": "Path does not exist",
                "path": str(root),
            },
            indent=2,
        )

    if not root.is_dir():
        return json.dumps(
            {
                "error": "Path is not a directory",
                "path": str(root),
            },
            indent=2,
        )

    candidates: list[dict[str, Any]] = []
    examined_files = 0

    for current_root, directories, files in os.walk(root):
        directories[:] = [
            directory
            for directory in directories
            if directory not in EXCLUDED_DIRECTORIES
        ]

        for filename in files:
            path = Path(current_root) / filename

            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue

            examined_files += 1
            content = read_small_text_file(path)

            if content is None:
                continue

            score, reasons = entry_point_score(path, content)

            if score <= 0:
                continue

            candidates.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(root)),
                    "score": score,
                    "reasons": reasons,
                }
            )

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["relative_path"],
        )
    )

    result = {
        "project_root": str(root),
        "examined_source_files": examined_files,
        "likely_entry_points": candidates[:max_results],
        "build_metadata": inspect_build_metadata(root),
    }

    if not candidates:
        result["message"] = (
            "No definite entry point was found. Check build metadata, "
            "generated sources or nonstandard application startup code."
        )

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )


def parse_temperature(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "temperature must be a number"
        ) from error

    if parsed < 0.0 or parsed > 2.0:
        raise argparse.ArgumentTypeError(
            "temperature must be between 0.0 and 2.0"
        )

    return parsed


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "max_tokens must be an integer"
        ) from error

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "max_tokens must be greater than 0"
        )

    return parsed


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge local MCP filesystem tools to an OpenAI-compatible "
            "llama-server endpoint."
        )
    )

    parser.add_argument(
        "--workspace",
        default=WORKSPACE,
        help=(
            "Allowed filesystem root for MCP filesystem server "
            f"(default: {WORKSPACE})"
        ),
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=(
            "Model name sent in chat payload "
            f"(default: {MODEL_NAME})"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=parse_temperature,
        default=TEMPERATURE,
        help=(
            "Sampling temperature sent in chat payload (0.0 to 2.0, "
            f"default: {TEMPERATURE})"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=parse_positive_int,
        default=MAX_TOKENS,
        help=(
            "max_tokens sent in chat payload "
            f"(default: {MAX_TOKENS})"
        ),
    )

    return parser.parse_args()


async def call_llama(
    client: httpx.AsyncClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
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
    if tool_name == "find_main_program":
        return find_main_program(arguments)

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
    return {
        "role": "user",
        "content": (
            f"The tool `{tool_name}` has completed.\n\n"
            "Tool arguments:\n"
            f"{json.dumps(arguments, ensure_ascii=False, indent=2)}\n\n"
            "Actual tool result:\n"
            f"{result_text}\n\n"
            "Answer the original user request using this result. "
            "Give the concrete paths and explain why each candidate is "
            "likely to be an entry point. Do not merely acknowledge the "
            "tool response and do not repeat the same tool call."
        ),
    }


async def run_agent(
    session: ClientSession,
    openai_tools: list[dict[str, Any]],
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> None:
    system_prompt = (
        "You are a local coding assistant with filesystem tools. "
        f"The only allowed filesystem root is `{WORKSPACE}`. "
        "Always use that exact absolute path or a path below it. "
        "Never use `/` as a filesystem path. "
        "When the user asks for the main program, application entry point, "
        "startup file or file containing main(), always call "
        "`find_main_program`. Do not guess programming languages or search "
        "only Python and JavaScript files. "
        "Use filesystem tools whenever information from files is needed. "
        "Prefer reading before editing. Never create, edit, move or delete "
        "files unless explicitly requested. "
        "After receiving a tool result, answer using the actual result."
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
            except (EOFError, KeyboardInterrupt):
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
                        model_name,
                        temperature,
                        max_tokens,
                    )
                except Exception as error:
                    print(
                        "\nModel request failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    break

                native_tool_calls = assistant.get("tool_calls") or []

                text_tool_calls: list[dict[str, Any]] = []

                if not native_tool_calls:
                    text_tool_calls = parse_text_tool_calls(
                        assistant.get("content") or ""
                    )

                tool_calls = native_tool_calls or text_tool_calls

                assistant_history: dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant.get("content") or "",
                }

                if native_tool_calls:
                    assistant_history["tool_calls"] = native_tool_calls

                messages.append(assistant_history)

                if not tool_calls:
                    answer = assistant.get("content") or "(empty response)"
                    print(f"\nAssistant: {answer}")
                    break

                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    tool_name = function.get("name")
                    call_id = tool_call.get("id")

                    if not isinstance(tool_name, str) or not tool_name:
                        print("\nInvalid tool call: missing tool name.")
                        continue

                    try:
                        arguments = parse_tool_arguments(
                            function.get("arguments")
                        )
                    except Exception as error:
                        print(
                            f"\nInvalid arguments for {tool_name}: {error}"
                        )
                        continue

                    print(
                        f"\n[tool {step + 1}] {tool_name}"
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
                    f"\nStopped after {MAX_AGENT_STEPS} tool-call steps."
                )


async def main() -> None:
    global WORKSPACE

    args = parse_cli_args()
    WORKSPACE = str(Path(args.workspace).resolve())

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
    print(f"model: {args.model}")
    print(f"temperature: {args.temperature}")
    print(f"max_tokens: {args.max_tokens}")

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

                openai_tools = (
                    mcp_tools_to_openai(available.tools)
                    + custom_tools()
                )

                print("Available tools:")

                for tool in openai_tools:
                    print(
                        "  - "
                        + tool["function"]["name"]
                    )

                await run_agent(
                    session,
                    openai_tools,
                    args.model,
                    args.temperature,
                    args.max_tokens,
                )

    except asyncio.CancelledError:
        return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")