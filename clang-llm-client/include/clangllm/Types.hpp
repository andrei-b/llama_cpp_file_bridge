#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace clangllm {

struct Position {
    int line = 0;
    int character = 0;
};

struct Range {
    Position start;
    Position end;
};

struct Symbol {
    std::filesystem::path file;
    std::string name;
    std::string detail;
    int kind = 0;
    Range range;
    Range selection_range;
};

struct Location {
    std::filesystem::path file;
    Range range;
    std::string container_name;
};

struct CallerContext {
    std::filesystem::path file;
    std::string caller_name;
    int reference_line = 0;
    std::string reference_excerpt;
    std::string caller_implementation;
};

struct FunctionContext {
    Symbol symbol;
    std::string qualified_name;
    std::string usr;
    std::string hover;
    std::string declaration_excerpt;
    std::string implementation;
    std::vector<CallerContext> callers;
    std::vector<std::string> diagnostics;
};

inline bool operator<(const Position& lhs, const Position& rhs) {
    return lhs.line < rhs.line ||
           (lhs.line == rhs.line && lhs.character < rhs.character);
}

inline bool operator<=(const Position& lhs, const Position& rhs) {
    return !(rhs < lhs);
}

inline bool contains(const Range& range, const Position& position) {
    return range.start <= position && position <= range.end;
}

} // namespace clangllm
