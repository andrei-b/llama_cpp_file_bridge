#pragma once

#include "clangllm/LspClient.hpp"
#include "clangllm/SourceStore.hpp"
#include "clangllm/Types.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace clangllm {

class ClangdAnalyzer {
public:
    ClangdAnalyzer(LspClient& lsp, SourceStore& sources,
                   std::filesystem::path project_root);

    std::vector<Symbol> functions_in_file(const std::filesystem::path& file);
    FunctionContext collect_context(const Symbol& function, std::size_t max_callers);

private:
    void open_file(const std::filesystem::path& file);
    std::vector<Symbol> load_symbols(const std::filesystem::path& file);
    std::optional<Symbol> enclosing_function(const std::filesystem::path& file,
                                             const Position& position);
    std::vector<Location> references(const Symbol& symbol);
    std::optional<Location> first_location_result(const boost::json::value& result);
    std::string hover_text(const boost::json::value& value) const;

    LspClient& lsp_;
    SourceStore& sources_;
    std::filesystem::path project_root_;
    std::unordered_set<std::string> opened_files_;
    std::unordered_map<std::string, std::vector<Symbol>> symbol_cache_;
};

} // namespace clangllm
