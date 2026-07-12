#include "clangllm/CompilationDatabase.hpp"

#include <boost/json.hpp>

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace clangllm {
namespace json = boost::json;

namespace {

bool is_cpp_translation_unit(const std::filesystem::path& file) {
    const auto ext = file.extension().string();
    return ext == ".c" || ext == ".cc" || ext == ".cpp" || ext == ".cxx" ||
           ext == ".C" || ext == ".m" || ext == ".mm";
}

std::string read_file(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot open compilation database: " + file.string());
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

} // namespace

CompilationDatabase::CompilationDatabase(const std::filesystem::path& build_directory) {
    const auto database_file = std::filesystem::absolute(build_directory) / "compile_commands.json";
    const auto parsed = json::parse(read_file(database_file));
    if (!parsed.is_array()) {
        throw std::runtime_error("compile_commands.json must contain a JSON array");
    }

    std::set<std::filesystem::path> unique;
    for (const auto& value : parsed.as_array()) {
        if (!value.is_object()) {
            continue;
        }
        const auto& object = value.as_object();
        const auto* directory_value = object.if_contains("directory");
        const auto* file_value = object.if_contains("file");
        if (directory_value == nullptr || file_value == nullptr ||
            !directory_value->is_string() || !file_value->is_string()) {
            continue;
        }

        std::filesystem::path file(file_value->as_string().c_str());
        if (file.is_relative()) {
            file = std::filesystem::path(directory_value->as_string().c_str()) / file;
        }
        file = std::filesystem::absolute(file).lexically_normal();

        if (is_cpp_translation_unit(file) && std::filesystem::exists(file)) {
            unique.insert(file);
        }
    }

    translation_units_.assign(unique.begin(), unique.end());
    if (translation_units_.empty()) {
        throw std::runtime_error("No source files found in " + database_file.string());
    }
}

} // namespace clangllm
