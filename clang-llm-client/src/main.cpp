#include "clangllm/ClangdAnalyzer.hpp"
#include "clangllm/CompilationDatabase.hpp"
#include "clangllm/LlmClient.hpp"
#include "clangllm/LspClient.hpp"
#include "clangllm/Prompt.hpp"
#include "clangllm/SourceStore.hpp"

#include <boost/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace json = boost::json;
using namespace clangllm;

namespace {

struct Options {
    std::filesystem::path project = ".";
    std::filesystem::path build = "build";
    std::string clangd = "clangd";
    std::string endpoint = "http://127.0.0.1:8080/v1/chat/completions";
    std::string model = "qwen2.5-coder";
    std::filesystem::path output = "clang-llm-report.jsonl";
    std::string name_contains;
    std::size_t max_functions = 0;
    std::size_t max_callers = 8;
    int index_wait_seconds = 5;
    bool dry_run = false;
};

[[noreturn]] void usage(const char* program, const std::string& error = {}) {
    if (!error.empty()) {
        std::cerr << "Error: " << error << "\n\n";
    }
    std::cerr
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --project PATH             Project root (default: .)\n"
        << "  --build PATH               Directory containing compile_commands.json\n"
        << "  --clangd PATH              clangd executable\n"
        << "  --endpoint URL             OpenAI-compatible /v1/chat/completions endpoint\n"
        << "  --model NAME               Local model name\n"
        << "  --output FILE              JSONL report path\n"
        << "  --name-contains TEXT       Analyze only matching function names\n"
        << "  --max-functions N          Stop after N functions; 0 means all\n"
        << "  --max-callers N            Maximum distinct caller contexts per function\n"
        << "  --index-wait-seconds N     Best-effort wait for clangd background index\n"
        << "  --dry-run                  Write context bundles without calling the LLM\n"
        << "  --help                     Show this help\n\n"
        << "Set CLANG_LLM_API_KEY if the endpoint requires a bearer token.\n";
    std::exit(error.empty() ? 0 : 2);
}

std::string require_value(int argc, char** argv, int& index, const std::string& option) {
    if (index + 1 >= argc) {
        usage(argv[0], "Missing value for " + option);
    }
    return argv[++index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            usage(argv[0]);
        } else if (argument == "--project") {
            options.project = require_value(argc, argv, i, argument);
        } else if (argument == "--build") {
            options.build = require_value(argc, argv, i, argument);
        } else if (argument == "--clangd") {
            options.clangd = require_value(argc, argv, i, argument);
        } else if (argument == "--endpoint") {
            options.endpoint = require_value(argc, argv, i, argument);
        } else if (argument == "--model") {
            options.model = require_value(argc, argv, i, argument);
        } else if (argument == "--output") {
            options.output = require_value(argc, argv, i, argument);
        } else if (argument == "--name-contains") {
            options.name_contains = require_value(argc, argv, i, argument);
        } else if (argument == "--max-functions") {
            options.max_functions = std::stoull(require_value(argc, argv, i, argument));
        } else if (argument == "--max-callers") {
            options.max_callers = std::stoull(require_value(argc, argv, i, argument));
        } else if (argument == "--index-wait-seconds") {
            options.index_wait_seconds = std::stoi(require_value(argc, argv, i, argument));
        } else if (argument == "--dry-run") {
            options.dry_run = true;
        } else {
            usage(argv[0], "Unknown option: " + argument);
        }
    }

    options.project = std::filesystem::absolute(options.project).lexically_normal();
    if (options.build.is_relative()) {
        options.build = options.project / options.build;
    }
    options.build = std::filesystem::absolute(options.build).lexically_normal();
    if (options.output.is_relative()) {
        options.output = options.project / options.output;
    }
    return options;
}

std::optional<std::string> api_key_from_environment() {
    if (const char* value = std::getenv("CLANG_LLM_API_KEY"); value != nullptr && *value != '\0') {
        return std::string(value);
    }
    return std::nullopt;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        CompilationDatabase compilation_database(options.build);

        std::cout << "Project: " << options.project << '\n'
                  << "Compilation database: " << options.build / "compile_commands.json" << '\n'
                  << "Translation units: " << compilation_database.translation_units().size() << '\n';

        LspClient lsp(options.project, options.build, options.clangd);
        lsp.start();
        lsp.initialize();

        if (options.index_wait_seconds > 0) {
            const bool complete = lsp.wait_for_background_index(
                std::chrono::seconds{options.index_wait_seconds});
            if (!complete) {
                std::cerr << "Warning: clangd background index did not report completion; "
                             "cross-file references may be incomplete on this run.\n";
            }
        }

        SourceStore sources;
        ClangdAnalyzer analyzer(lsp, sources, options.project);
        std::optional<LlmClient> llm;
        if (!options.dry_run) {
            llm.emplace(options.endpoint, options.model, api_key_from_environment());
        }

        std::ofstream report(options.output, std::ios::out | std::ios::trunc);
        if (!report) {
            throw std::runtime_error("Cannot open output file: " + options.output.string());
        }

        std::size_t analyzed = 0;
        std::size_t failures = 0;
        bool limit_reached = false;

        for (const auto& file : compilation_database.translation_units()) {
            std::vector<Symbol> functions;
            try {
                functions = analyzer.functions_in_file(file);
            } catch (const std::exception& error) {
                std::cerr << "Cannot enumerate functions in " << file << ": "
                          << error.what() << '\n';
                ++failures;
                continue;
            }

            for (const auto& function : functions) {
                if (!options.name_contains.empty() &&
                    function.name.find(options.name_contains) == std::string::npos) {
                    continue;
                }
                if (options.max_functions != 0 && analyzed >= options.max_functions) {
                    limit_reached = true;
                    break;
                }

                std::cout << "Analyzing " << file.filename().string() << ':'
                          << (function.range.start.line + 1) << " " << function.name << '\n';

                json::object record{
                    {"file", function.file.string()},
                    {"line", function.range.start.line + 1},
                    {"symbol", function.name},
                };

                try {
                    const auto context = analyzer.collect_context(function, options.max_callers);
                    record["qualified_name"] = context.qualified_name;
                    record["usr"] = context.usr;

                    if (options.dry_run) {
                        record["context"] = context_to_json(context);
                    } else {
                        record["review"] = llm->chat(build_review_messages(context));
                    }
                } catch (const std::exception& error) {
                    record["error"] = error.what();
                    ++failures;
                    std::cerr << "  failed: " << error.what() << '\n';
                }

                report << json::serialize(record) << '\n';
                report.flush();
                ++analyzed;
            }

            if (limit_reached) {
                break;
            }
        }

        lsp.stop();
        std::cout << "Analyzed functions: " << analyzed << '\n'
                  << "Failures: " << failures << '\n'
                  << "Report: " << options.output << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        return 2;
    }
}
