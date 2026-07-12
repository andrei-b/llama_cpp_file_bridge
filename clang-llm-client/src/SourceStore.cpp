#include "clangllm/SourceStore.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace clangllm {

const std::string& SourceStore::text(const std::filesystem::path& input) {
    const auto file = std::filesystem::absolute(input).lexically_normal();
    const auto key = file.string();

    std::scoped_lock lock(mutex_);
    if (auto it = cache_.find(key); it != cache_.end()) {
        return it->second;
    }

    std::ifstream stream(file, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot open source file: " + file.string());
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return cache_.emplace(key, buffer.str()).first->second;
}

std::size_t SourceStore::offset_for_position(const std::string& source, const Position& position) {
    if (position.line < 0 || position.character < 0) {
        return 0;
    }

    std::size_t offset = 0;
    int line = 0;
    while (line < position.line && offset < source.size()) {
        const auto newline = source.find('\n', offset);
        if (newline == std::string::npos) {
            return source.size();
        }
        offset = newline + 1;
        ++line;
    }

    return std::min(source.size(), offset + static_cast<std::size_t>(position.character));
}

std::string SourceStore::extract(const std::filesystem::path& file, const Range& range) {
    const auto& source = text(file);
    const auto begin = offset_for_position(source, range.start);
    const auto end = offset_for_position(source, range.end);
    if (end < begin) {
        return {};
    }
    return source.substr(begin, end - begin);
}

std::string SourceStore::line_window(const std::filesystem::path& file, int center_line,
                                     int before, int after) {
    const auto& source = text(file);
    std::istringstream input(source);
    std::ostringstream output;
    std::string line;
    int line_number = 0;
    const int first = std::max(0, center_line - before);
    const int last = center_line + after;

    while (std::getline(input, line)) {
        if (line_number >= first && line_number <= last) {
            output << (line_number + 1) << ": " << line << '\n';
        }
        if (line_number > last) {
            break;
        }
        ++line_number;
    }
    return output.str();
}

void SourceStore::invalidate(const std::filesystem::path& input) {
    const auto key = std::filesystem::absolute(input).lexically_normal().string();
    std::scoped_lock lock(mutex_);
    cache_.erase(key);
}

std::string truncate_middle(std::string text, std::size_t max_chars) {
    if (text.size() <= max_chars || max_chars < 64) {
        return text;
    }

    const std::size_t marker_space = 35;
    const std::size_t side = (max_chars - marker_space) / 2;
    return text.substr(0, side) +
           "\n/* ... context truncated ... */\n" +
           text.substr(text.size() - side);
}

} // namespace clangllm
