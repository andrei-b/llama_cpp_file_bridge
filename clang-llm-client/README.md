# clang-llm client (MVP)

A read-only C++ analyzer that combines:

- `compile_commands.json` to discover translation units;
- a long-running `clangd` process for semantic symbols, declarations, references, hover information, and diagnostics;
- an OpenAI-compatible local LLM endpoint such as `llama-server` or Ollama;
- JSONL output containing one review per function or method.

The tool does **not** edit source code. Its first goal is to produce conservative, reviewable findings and suggested tests.

## What it does

For every function or method definition found in the compilation database, the program collects:

1. the implementation body;
2. the qualified symbol identity and Clang USR when available;
3. hover information and documentation;
4. the declaration context;
5. up to N distinct enclosing functions that reference it, with the reference line highlighted;
6. current clang/clang-tidy diagnostics published by clangd.

It sends that context to the LLM and asks it to compare observed implementation behavior with contract evidence from declarations, types, comments, callers, tests, assertions, and diagnostics.

## Requirements

Ubuntu/Debian packages:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake pkg-config \
  clangd libboost-dev libcurl4-openssl-dev
```

Your project must have a compilation database:

```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## Build

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

## Run a context-only test

This checks the clangd integration without calling an LLM:

```bash
./build/clang-llm \
  --project /path/to/project \
  --build /path/to/project/build \
  --dry-run \
  --max-functions 5 \
  --output /tmp/clang-llm-context.jsonl
```

## Run with llama.cpp

Start an OpenAI-compatible server, for example:

```bash
llama-server \
  -m /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 16384
```

Then run:

```bash
./build/clang-llm \
  --project /path/to/project \
  --build /path/to/project/build \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --model local-coder \
  --max-functions 20 \
  --max-callers 8
```

For Ollama's OpenAI-compatible endpoint, use a URL such as:

```bash
--endpoint http://127.0.0.1:11434/v1/chat/completions
```

If authentication is required:

```bash
export CLANG_LLM_API_KEY='...'
```

## Useful filters

Analyze one family of functions first:

```bash
./build/clang-llm \
  --project . \
  --build build \
  --name-contains connect \
  --max-functions 10
```

Set `--max-functions 0` to analyze all discovered definitions.

## Output

Each JSONL line contains the source location, symbol identity, and either:

- `context`, in `--dry-run` mode;
- `review`, containing the parsed model JSON;
- `error`, if one function could not be processed.

The expected review shape is:

```json
{
  "observed_behavior": "...",
  "inferred_contract": ["..."],
  "findings": [
    {
      "severity": "warning",
      "confidence": 0.86,
      "category": "error handling",
      "summary": "...",
      "evidence": ["..."],
      "suggested_test": "..."
    }
  ],
  "uncertainties": ["..."]
}
```

## Important limitations of this MVP

- A clangd reference is not necessarily a direct call. It may be a declaration, callback registration, or address-taking expression. The prompt explicitly warns the model about this.
- Virtual dispatch, function pointers, macros, generated sources, and some template relationships require additional handling.
- The first run may have incomplete cross-file references if clangd's background index has not finished. Subsequent runs benefit from clangd's cached index.
- Reviewing every function independently is expensive and repeats context. A production version should cache function summaries by source hash, compile-command hash, model, and prompt version.
- Caller assumptions do not fully define intent. Tests, interface declarations, documentation, invariants, and runtime evidence should be included whenever possible.
- LLM findings are hypotheses. Confirm them using focused tests, sanitizers, compilation, clang-tidy, and static analysis before treating them as defects.

## Recommended next iterations

1. Add a LibTooling pass that classifies direct calls, member calls, callback registrations, overrides, reads/writes of fields, and return-value usage.
2. Use a two-pass workflow: cache an observed-behavior summary per function, then compare caller assumptions against those compact summaries.
3. Add Git-diff mode so pull-request reviews analyze changed functions plus one or two levels of callers/callees.
4. Add SQLite caching and token budgets.
5. Add verification commands for build, clang-tidy, tests, and sanitizers.
6. Expose the analyzer as an MCP server for the local LLM agent.
