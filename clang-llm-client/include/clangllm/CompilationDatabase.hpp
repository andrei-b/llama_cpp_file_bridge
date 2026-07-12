#pragma once

#include <filesystem>
#include <vector>

namespace clangllm {

class CompilationDatabase {
public:
    explicit CompilationDatabase(const std::filesystem::path& build_directory);

    const std::vector<std::filesystem::path>& translation_units() const noexcept {
        return translation_units_;
    }

private:
    std::vector<std::filesystem::path> translation_units_;
};

} // namespace clangllm
