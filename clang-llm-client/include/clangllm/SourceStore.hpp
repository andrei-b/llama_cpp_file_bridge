#pragma once

#include "clangllm/Types.hpp"

#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

namespace clangllm {

class SourceStore {
public:
    const std::string& text(const std::filesystem::path& file);
    std::string extract(const std::filesystem::path& file, const Range& range);
    std::string line_window(const std::filesystem::path& file, int center_line,
                            int before, int after);
    void invalidate(const std::filesystem::path& file);

private:
    static std::size_t offset_for_position(const std::string& text, const Position& position);

    std::mutex mutex_;
    std::unordered_map<std::string, std::string> cache_;
};

std::string truncate_middle(std::string text, std::size_t max_chars);

} // namespace clangllm
