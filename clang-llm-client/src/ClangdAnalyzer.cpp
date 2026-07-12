#include "clangllm/ClangdAnalyzer.hpp"
#include "clangllm/Uri.hpp"

#include <boost/json.hpp>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace clangllm {
namespace json = boost::json;

namespace {

Position parse_position(const json::value& value) {
    const auto& object = value.as_object();
    return Position{
        static_cast<int>(object.at("line").as_int64()),
        static_cast<int>(object.at("character").as_int64()),
    };
}

Range parse_range(const json::value& value) {
    const auto& object = value.as_object();
    return Range{parse_position(object.at("start")), parse_position(object.at("end"))};
}

json::object lsp_position(const Position& position) {
    return json::object{{"line", position.line}, {"character", position.character}};
}


bool is_function_kind(int kind) {
    // LSP SymbolKind: Method=6, Constructor=9, Function=12.
    return kind == 6 || kind == 9 || kind == 12;
}

std::size_t range_span(const Range& range) {
    const auto lines = static_cast<std::size_t>(std::max(0, range.end.line - range.start.line));
    const auto columns = static_cast<std::size_t>(std::max(0, range.end.character - range.start.character));
    return lines * 1'000'000U + columns;
}

bool is_test_path(const std::filesystem::path& path) {
    std::string text = path.generic_string();
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text.find("/test/") != std::string::npos ||
           text.find("/tests/") != std::string::npos ||
           text.find("_test.") != std::string::npos ||
           text.find("test_") != std::string::npos;
}

bool path_is_within(const std::filesystem::path& path, const std::filesystem::path& root) {
    const auto normalized_path = std::filesystem::absolute(path).lexically_normal();
    const auto normalized_root = std::filesystem::absolute(root).lexically_normal();
    auto pit = normalized_path.begin();
    auto rit = normalized_root.begin();
    for (; rit != normalized_root.end(); ++rit, ++pit) {
        if (pit == normalized_path.end() || *pit != *rit) {
            return false;
        }
    }
    return true;
}

void append_document_symbols(const json::array& array,
                             const std::filesystem::path& file,
                             std::vector<Symbol>& output) {
    for (const auto& value : array) {
        if (!value.is_object()) {
            continue;
        }
        const auto& object = value.as_object();
        const auto* name = object.if_contains("name");
        const auto* kind = object.if_contains("kind");
        if (name == nullptr || kind == nullptr || !name->is_string() || !kind->is_int64()) {
            continue;
        }

        Symbol symbol;
        symbol.file = file;
        symbol.name = name->as_string().c_str();
        symbol.kind = static_cast<int>(kind->as_int64());
        if (const auto* detail = object.if_contains("detail"); detail != nullptr && detail->is_string()) {
            symbol.detail = detail->as_string().c_str();
        }

        if (const auto* range = object.if_contains("range"); range != nullptr && range->is_object()) {
            symbol.range = parse_range(*range);
            if (const auto* selection = object.if_contains("selectionRange");
                selection != nullptr && selection->is_object()) {
                symbol.selection_range = parse_range(*selection);
            } else {
                symbol.selection_range = symbol.range;
            }
            output.push_back(std::move(symbol));
        } else if (const auto* location = object.if_contains("location");
                   location != nullptr && location->is_object()) {
            const auto& location_object = location->as_object();
            if (const auto* range = location_object.if_contains("range");
                range != nullptr && range->is_object()) {
                symbol.range = parse_range(*range);
                symbol.selection_range = symbol.range;
                output.push_back(std::move(symbol));
            }
        }

        if (const auto* children = object.if_contains("children");
            children != nullptr && children->is_array()) {
            append_document_symbols(children->as_array(), file, output);
        }
    }
}

std::optional<Location> parse_location(const json::value& value) {
    if (!value.is_object()) {
        return std::nullopt;
    }
    const auto& object = value.as_object();

    const auto* uri = object.if_contains("uri");
    const auto* range = object.if_contains("range");
    if (uri != nullptr && range != nullptr && uri->is_string() && range->is_object()) {
        Location result;
        result.file = uri_to_path(uri->as_string().c_str());
        result.range = parse_range(*range);
        if (const auto* container = object.if_contains("containerName");
            container != nullptr && container->is_string()) {
            result.container_name = container->as_string().c_str();
        }
        return result;
    }

    const auto* target_uri = object.if_contains("targetUri");
    const auto* target_range = object.if_contains("targetRange");
    if (target_uri != nullptr && target_range != nullptr &&
        target_uri->is_string() && target_range->is_object()) {
        return Location{uri_to_path(target_uri->as_string().c_str()), parse_range(*target_range), {}};
    }
    return std::nullopt;
}

} // namespace

ClangdAnalyzer::ClangdAnalyzer(LspClient& lsp, SourceStore& sources,
                               std::filesystem::path project_root)
    : lsp_(lsp), sources_(sources),
      project_root_(std::filesystem::absolute(std::move(project_root)).lexically_normal()) {}

void ClangdAnalyzer::open_file(const std::filesystem::path& input) {
    const auto file = std::filesystem::absolute(input).lexically_normal();
    const auto key = file.string();
    if (!opened_files_.insert(key).second) {
        return;
    }

    json::object document{
        {"uri", path_to_uri(file)},
        {"languageId", file.extension() == ".c" ? "c" : "cpp"},
        {"version", 1},
        {"text", sources_.text(file)},
    };
    lsp_.notify("textDocument/didOpen",
                json::object{{"textDocument", std::move(document)}});
}

std::vector<Symbol> ClangdAnalyzer::load_symbols(const std::filesystem::path& input) {
    const auto file = std::filesystem::absolute(input).lexically_normal();
    const auto key = file.string();
    if (const auto it = symbol_cache_.find(key); it != symbol_cache_.end()) {
        return it->second;
    }

    open_file(file);
    const auto result = lsp_.request(
        "textDocument/documentSymbol",
        json::object{{"textDocument", json::object{{"uri", path_to_uri(file)}}}});

    std::vector<Symbol> symbols;
    if (result.is_array()) {
        append_document_symbols(result.as_array(), file, symbols);
    }
    symbol_cache_[key] = symbols;
    return symbols;
}

std::vector<Symbol> ClangdAnalyzer::functions_in_file(const std::filesystem::path& file) {
    std::vector<Symbol> functions;
    for (auto& symbol : load_symbols(file)) {
        if (!is_function_kind(symbol.kind)) {
            continue;
        }

        const std::string source = sources_.extract(symbol.file, symbol.range);
        const bool has_open_brace = source.find('{') != std::string::npos;
        const bool has_close_brace = source.rfind('}') != std::string::npos;
        const bool deleted_or_defaulted = source.find("= delete") != std::string::npos ||
                                          source.find("= default") != std::string::npos;
        if (has_open_brace && has_close_brace && !deleted_or_defaulted) {
            functions.push_back(std::move(symbol));
        }
    }
    return functions;
}

std::optional<Symbol> ClangdAnalyzer::enclosing_function(const std::filesystem::path& file,
                                                         const Position& position) {
    std::optional<Symbol> best;
    for (const auto& symbol : functions_in_file(file)) {
        if (!contains(symbol.range, position)) {
            continue;
        }
        if (!best || range_span(symbol.range) < range_span(best->range)) {
            best = symbol;
        }
    }
    return best;
}

std::vector<Location> ClangdAnalyzer::references(const Symbol& symbol) {
    open_file(symbol.file);
    const auto result = lsp_.request(
        "textDocument/references",
        json::object{
            {"textDocument", json::object{{"uri", path_to_uri(symbol.file)}}},
            {"position", lsp_position(symbol.selection_range.start)},
            {"context", json::object{{"includeDeclaration", true}}},
        });

    std::vector<Location> locations;
    if (!result.is_array()) {
        return locations;
    }
    for (const auto& item : result.as_array()) {
        if (auto location = parse_location(item)) {
            locations.push_back(std::move(*location));
        }
    }
    return locations;
}

std::optional<Location> ClangdAnalyzer::first_location_result(const json::value& result) {
    if (auto location = parse_location(result)) {
        return location;
    }
    if (result.is_array()) {
        for (const auto& item : result.as_array()) {
            if (auto location = parse_location(item)) {
                return location;
            }
        }
    }
    return std::nullopt;
}

std::string ClangdAnalyzer::hover_text(const json::value& value) const {
    if (value.is_null()) {
        return {};
    }
    if (value.is_string()) {
        return value.as_string().c_str();
    }
    if (value.is_array()) {
        std::ostringstream output;
        for (const auto& item : value.as_array()) {
            const auto text = hover_text(item);
            if (!text.empty()) output << text << '\n';
        }
        return output.str();
    }
    if (!value.is_object()) {
        return json::serialize(value);
    }

    const auto& object = value.as_object();
    if (const auto* contents = object.if_contains("contents")) {
        return hover_text(*contents);
    }
    if (const auto* value_field = object.if_contains("value");
        value_field != nullptr && value_field->is_string()) {
        return value_field->as_string().c_str();
    }
    return json::serialize(value);
}

FunctionContext ClangdAnalyzer::collect_context(const Symbol& function, std::size_t max_callers) {
    open_file(function.file);
    FunctionContext context;
    context.symbol = function;
    context.implementation = truncate_middle(sources_.extract(function.file, function.range), 10'000);

    const json::object document_position{
        {"textDocument", json::object{{"uri", path_to_uri(function.file)}}},
        {"position", lsp_position(function.selection_range.start)},
    };

    try {
        const auto symbol_info = lsp_.request("textDocument/symbolInfo", document_position);
        if (symbol_info.is_array() && !symbol_info.as_array().empty() &&
            symbol_info.as_array().front().is_object()) {
            const auto& info = symbol_info.as_array().front().as_object();
            const auto* name = info.if_contains("name");
            const auto* container = info.if_contains("containerName");
            const std::string simple_name = name != nullptr && name->is_string()
                                                ? std::string(name->as_string().c_str())
                                                : function.name;
            const std::string container_name = container != nullptr && container->is_string()
                                                   ? std::string(container->as_string().c_str())
                                                   : std::string{};
            context.qualified_name = container_name.empty()
                                         ? simple_name
                                         : container_name + "::" + simple_name;
            if (const auto* usr = info.if_contains("usr"); usr != nullptr && usr->is_string()) {
                context.usr = usr->as_string().c_str();
            }
        }
    } catch (const std::exception&) {
        context.qualified_name = function.name;
    }

    if (context.qualified_name.empty()) {
        context.qualified_name = function.name;
    }

    try {
        context.hover = hover_text(lsp_.request("textDocument/hover", document_position));
    } catch (const std::exception&) {
        // Hover is useful but optional.
    }

    try {
        const auto declaration_result = lsp_.request("textDocument/declaration", document_position);
        if (auto declaration = first_location_result(declaration_result)) {
            if (path_is_within(declaration->file, project_root_) &&
                std::filesystem::exists(declaration->file)) {
                context.declaration_excerpt =
                    sources_.line_window(declaration->file, declaration->range.start.line, 5, 8);
            }
        }
    } catch (const std::exception&) {
        // A separate declaration does not always exist.
    }

    auto locations = references(function);
    std::stable_sort(locations.begin(), locations.end(), [](const Location& lhs, const Location& rhs) {
        return is_test_path(lhs.file) && !is_test_path(rhs.file);
    });

    std::set<std::string> seen_callers;
    for (const auto& location : locations) {
        if (context.callers.size() >= max_callers) {
            break;
        }
        if (!path_is_within(location.file, project_root_) || !std::filesystem::exists(location.file)) {
            continue;
        }
        if (std::filesystem::equivalent(location.file, function.file) &&
            contains(function.range, location.range.start)) {
            continue;
        }

        try {
            auto caller = enclosing_function(location.file, location.range.start);
            if (!caller) {
                continue;
            }
            const std::string key = caller->file.string() + ":" +
                                    std::to_string(caller->range.start.line) + ":" +
                                    std::to_string(caller->range.start.character);
            if (!seen_callers.insert(key).second) {
                continue;
            }

            CallerContext caller_context;
            caller_context.file = caller->file;
            caller_context.caller_name = caller->name;
            caller_context.reference_line = location.range.start.line + 1;
            caller_context.reference_excerpt =
                sources_.line_window(location.file, location.range.start.line, 3, 3);
            caller_context.caller_implementation =
                truncate_middle(sources_.extract(caller->file, caller->range), 4'000);
            context.callers.push_back(std::move(caller_context));
        } catch (const std::exception& error) {
            std::cerr << "Skipping reference context in " << location.file << ": "
                      << error.what() << '\n';
        }
    }

    context.diagnostics = lsp_.diagnostics_for_uri(path_to_uri(function.file));
    return context;
}

} // namespace clangllm
