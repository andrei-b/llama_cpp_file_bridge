#pragma once

#include <filesystem>
#include <string>

namespace clangllm {

std::string path_to_uri(const std::filesystem::path& path);
std::filesystem::path uri_to_path(const std::string& uri);

} // namespace clangllm
