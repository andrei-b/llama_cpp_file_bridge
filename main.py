#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


LLAMA_URL = os.environ.get(
    "LLAMA_URL",
    "http://127.0.0.1:8080/v1/chat/completions",
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

    def tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    path_property = {
        "type": "string",
        "description": "Absolute path inside the allowed workspace.",
    }

    return [
        tool(
            "find_main_program",
            "Find likely program entry-point files in a software project.",
            {
                "path": path_property,
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 15},
            },
            ["path"],
        ),
        tool(
            "read_lines",
            "Read an inclusive range of lines from a text file and return numbered lines.",
            {
                "path": path_property,
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            ["path", "start_line", "end_line"],
        ),
        tool(
            "list_symbols",
            "List likely classes, structs, namespaces and functions declared in a source file.",
            {"path": path_property},
            ["path"],
        ),
        tool(
            "find_definition",
            "Find likely definitions or declarations of a symbol in the workspace.",
            {
                "symbol": {"type": "string"},
                "path": path_property,
                "max_results": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
            },
            ["symbol"],
        ),
        tool(
            "find_references",
            "Find textual references to a symbol in source files in the workspace.",
            {
                "symbol": {"type": "string"},
                "path": path_property,
                "max_results": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
            },
            ["symbol"],
        ),
        tool(
            "find_corresponding_header",
            "Find likely header files corresponding to a C or C++ source file.",
            {"path": path_property},
            ["path"],
        ),
        tool(
            "find_corresponding_source",
            "Find likely C or C++ source files corresponding to a header file.",
            {"path": path_property},
            ["path"],
        ),
        tool(
            "extract_containing_function",
            "Extract the function or method containing a given 1-based source line.",
            {
                "path": path_property,
                "line": {"type": "integer", "minimum": 1},
            },
            ["path", "line"],
        ),
        tool(
            "run_clang_tidy",
            "Run clang-tidy on a C or C++ source file and return diagnostics.",
            {
                "path": path_property,
                "build_path": path_property,
                "checks": {"type": "string", "default": ""},
                "fix": {"type": "boolean", "default": False},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 1800, "default": 300},
            },
            ["path"],
        ),
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



def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _resolve_allowed_path(value: Any, default: str | None = None) -> Path:
    if value is None:
        value = default if default is not None else WORKSPACE
    if not isinstance(value, str):
        raise ValueError("path must be a string")
    path = Path(value).expanduser().resolve()
    if not is_path_allowed(str(path)):
        raise ValueError(f"path is outside allowed workspace: {path}")
    return path


def _iter_source_files(root: Path):
    if root.is_file():
        if root.suffix.lower() in SOURCE_EXTENSIONS:
            yield root
        return
    for current_root, directories, files in os.walk(root):
        directories[:] = [d for d in directories if d not in EXCLUDED_DIRECTORIES]
        for name in files:
            path = Path(current_root) / name
            if path.suffix.lower() in SOURCE_EXTENSIONS:
                yield path


def read_lines_tool(arguments: dict[str, Any]) -> str:
    try:
        path = _resolve_allowed_path(arguments.get("path"))
        start = int(arguments.get("start_line"))
        end = int(arguments.get("end_line"))
        if start < 1 or end < start:
            raise ValueError("require 1 <= start_line <= end_line")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = [f"{i:6d}: {lines[i-1]}" for i in range(start, min(end, len(lines)) + 1)]
        return _json({"path": str(path), "start_line": start, "end_line": min(end, len(lines)), "total_lines": len(lines), "text": "\n".join(selected)})
    except Exception as error:
        return _json({"error": str(error)})


def list_symbols_tool(arguments: dict[str, Any]) -> str:
    try:
        path = _resolve_allowed_path(arguments.get("path"))
        text = path.read_text(encoding="utf-8", errors="replace")
        symbols: list[dict[str, Any]] = []
        patterns = [
            ("namespace", re.compile(r"^\s*namespace\s+([A-Za-z_]\w*)", re.M)),
            ("class", re.compile(r"^\s*(?:template\s*<[^;{]+>\s*)?(class|struct|enum(?:\s+class)?)\s+([A-Za-z_]\w*)", re.M)),
            ("function", re.compile(r"^\s*(?:template\s*<[^;{]+>\s*)?(?:[\w:\<\>\*&\s]+?)\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\([^;{}]*\)\s*(?:const\s*)?(?:noexcept(?:\([^)]*\))?\s*)?(?:->\s*[^\{]+)?\s*\{", re.M)),
            ("python_function", re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", re.M)),
            ("python_class", re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*(?:\(|:)", re.M)),
        ]
        for kind, pattern in patterns:
            for match in pattern.finditer(text):
                name = match.group(2) if kind == "class" else match.group(1)
                line = text.count("\n", 0, match.start()) + 1
                symbols.append({"name": name, "kind": kind, "line": line})
        symbols.sort(key=lambda x: (x["line"], x["name"]))
        return _json({"path": str(path), "symbols": symbols})
    except Exception as error:
        return _json({"error": str(error)})


def _search_symbol(symbol: str, root: Path, max_results: int, definitions_only: bool) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    escaped = re.escape(symbol)
    if definitions_only:
        regex = re.compile(rf"(?:\bclass\s+|\bstruct\s+|\benum(?:\s+class)?\s+|\bdef\s+|\bfn\s+|\bfunc\s+|\b(?:[\w:<>,~*&]+\s+)+){escaped}\s*(?:\(|\{{|:)")
    else:
        regex = re.compile(rf"\b{escaped}\b")
    for path in _iter_source_files(root):
        content = read_small_text_file(path)
        if content is None:
            continue
        for line_no, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                results.append({"path": str(path), "line": line_no, "text": line.strip()})
                if len(results) >= max_results:
                    return results
    return results


def find_definition_tool(arguments: dict[str, Any]) -> str:
    try:
        symbol = arguments.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        root = _resolve_allowed_path(arguments.get("path"), WORKSPACE)
        limit = max(1, min(int(arguments.get("max_results", 20)), 100))
        return _json({"symbol": symbol, "results": _search_symbol(symbol, root, limit, True)})
    except Exception as error:
        return _json({"error": str(error)})


def find_references_tool(arguments: dict[str, Any]) -> str:
    try:
        symbol = arguments.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        root = _resolve_allowed_path(arguments.get("path"), WORKSPACE)
        limit = max(1, min(int(arguments.get("max_results", 100)), 500))
        return _json({"symbol": symbol, "results": _search_symbol(symbol, root, limit, False)})
    except Exception as error:
        return _json({"error": str(error)})


def _counterpart_candidates(path: Path, extensions: tuple[str, ...]) -> list[dict[str, Any]]:
    stem = path.stem
    candidates: list[Path] = []
    preferred_dirs = [path.parent, Path(WORKSPACE) / "include", Path(WORKSPACE) / "src"]
    for directory in preferred_dirs:
        for ext in extensions:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists() and candidate.resolve() != path.resolve():
                candidates.append(candidate.resolve())
    for candidate in Path(WORKSPACE).rglob(f"{stem}.*"):
        if any(part in EXCLUDED_DIRECTORIES for part in candidate.parts):
            continue
        if candidate.suffix.lower() in extensions and candidate.resolve() != path.resolve():
            candidates.append(candidate.resolve())
    unique=[]; seen=set()
    for candidate in candidates:
        key=str(candidate)
        if key not in seen:
            seen.add(key); unique.append({"path": key, "relative_path": str(candidate.relative_to(Path(WORKSPACE)))})
    return unique


def find_corresponding_header_tool(arguments: dict[str, Any]) -> str:
    try:
        path = _resolve_allowed_path(arguments.get("path"))
        return _json({"source": str(path), "headers": _counterpart_candidates(path, (".h", ".hh", ".hpp", ".hxx"))})
    except Exception as error:
        return _json({"error": str(error)})


def find_corresponding_source_tool(arguments: dict[str, Any]) -> str:
    try:
        path = _resolve_allowed_path(arguments.get("path"))
        return _json({"header": str(path), "sources": _counterpart_candidates(path, (".c", ".cc", ".cpp", ".cxx"))})
    except Exception as error:
        return _json({"error": str(error)})


def _brace_matching_end(text: str, opening: int) -> int | None:
    depth=0; state="code"; i=opening
    while i < len(text):
        c=text[i]; n=text[i+1] if i+1 < len(text) else ""
        if state == "code":
            if c == '"': state="string"
            elif c == "'": state="char"
            elif c == "/" and n == "/": state="line_comment"; i += 1
            elif c == "/" and n == "*": state="block_comment"; i += 1
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0: return i
        elif state == "string":
            if c == "\\": i += 1
            elif c == '"': state="code"
        elif state == "char":
            if c == "\\": i += 1
            elif c == "'": state="code"
        elif state == "line_comment":
            if c == "\n": state="code"
        elif state == "block_comment":
            if c == "*" and n == "/": state="code"; i += 1
        i += 1
    return None


def extract_containing_function_tool(arguments: dict[str, Any]) -> str:
    try:
        path = _resolve_allowed_path(arguments.get("path"))
        target_line = int(arguments.get("line"))
        text = path.read_text(encoding="utf-8", errors="replace")
        if target_line < 1 or target_line > text.count("\n") + 1:
            raise ValueError("line is outside file")
        function_pattern = re.compile(r"(?m)^[ \t]*(?:template\s*<[^;{]+>\s*)?(?:[\w:\<\>\*&\[\],~]+[ \t]+)+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\([^;{}]*\)\s*(?:const\s*)?(?:noexcept(?:\([^)]*\))?\s*)?(?:->\s*[^\{]+)?\s*\{")
        candidates=[]
        for match in function_pattern.finditer(text):
            opening=text.find("{", match.start(), match.end())
            end=_brace_matching_end(text, opening)
            if end is None: continue
            start_line=text.count("\n", 0, match.start())+1
            end_line=text.count("\n", 0, end)+1
            if start_line <= target_line <= end_line:
                candidates.append((start_line, end_line, match.group(1), match.start(), end+1))
        if not candidates:
            return _json({"path": str(path), "line": target_line, "found": False})
        start_line,end_line,name,start_pos,end_pos=max(candidates, key=lambda c:c[0])
        return _json({"path": str(path), "line": target_line, "found": True, "symbol": name, "start_line": start_line, "end_line": end_line, "code": text[start_pos:end_pos]})
    except Exception as error:
        return _json({"error": str(error)})


def _find_compile_db(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        direct=candidate / "compile_commands.json"
        build=candidate / "build" / "compile_commands.json"
        if direct.exists(): return candidate
        if build.exists(): return candidate / "build"
        if candidate == Path(WORKSPACE): break
    for path in Path(WORKSPACE).rglob("compile_commands.json"):
        if not any(part in EXCLUDED_DIRECTORIES - {"build", "cmake-build-debug", "cmake-build-release"} for part in path.parts):
            return path.parent
    return None


def run_clang_tidy_tool(arguments: dict[str, Any]) -> str:
    try:
        executable=shutil.which("clang-tidy")
        if executable is None:
            raise ValueError("clang-tidy is not installed or not in PATH")
        path=_resolve_allowed_path(arguments.get("path"))
        if path.suffix.lower() not in {".c", ".cc", ".cpp", ".cxx"}:
            raise ValueError("clang-tidy should be run on a C/C++ source file")
        build_value=arguments.get("build_path")
        build_path=_resolve_allowed_path(build_value) if build_value else _find_compile_db(path.parent)
        if build_path is None:
            raise ValueError("compile_commands.json not found; configure CMake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
        checks=arguments.get("checks", "")
        fix=bool(arguments.get("fix", False))
        timeout=max(1, min(int(arguments.get("timeout_seconds", 300)), 1800))
        command=[executable, str(path), "-p", str(build_path)]
        if isinstance(checks, str) and checks: command.append(f"-checks={checks}")
        if fix: command.append("--fix")
        completed=subprocess.run(command, cwd=WORKSPACE, capture_output=True, text=True, timeout=timeout, check=False)
        return _json({"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "build_path": str(build_path)})
    except subprocess.TimeoutExpired as error:
        return _json({"error": f"clang-tidy timed out after {error.timeout} seconds"})
    except Exception as error:
        return _json({"error": str(error)})

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
    custom_dispatch = {
        "find_main_program": find_main_program,
        "read_lines": read_lines_tool,
        "list_symbols": list_symbols_tool,
        "find_definition": find_definition_tool,
        "find_references": find_references_tool,
        "find_corresponding_header": find_corresponding_header_tool,
        "find_corresponding_source": find_corresponding_source_tool,
        "extract_containing_function": extract_containing_function_tool,
        "run_clang_tidy": run_clang_tidy_tool,
    }
    if tool_name in custom_dispatch:
        return custom_dispatch[tool_name](arguments)

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
        "Use code-intelligence tools such as read_lines, list_symbols, "
        "find_definition, find_references, corresponding-file lookup, "
        "extract_containing_function and run_clang_tidy when appropriate. "
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