#include "clangllm/Uri.hpp"

#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace clangllm {
namespace {

bool is_unreserved(unsigned char c) {
    return std::isalnum(c) != 0 || c == '-' || c == '_' || c == '.' || c == '~' || c == '/';
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

} // namespace

std::string path_to_uri(const std::filesystem::path& input) {
    const auto path = std::filesystem::absolute(input).lexically_normal().generic_string();
    std::ostringstream out;
    out << "file://";

    for (unsigned char c : path) {
        if (is_unreserved(c)) {
            out << static_cast<char>(c);
        } else {
            out << '%' << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(c) << std::nouppercase << std::dec;
        }
    }
    return out.str();
}

std::filesystem::path uri_to_path(const std::string& uri) {
    constexpr const char* prefix = "file://";
    if (!uri.starts_with(prefix)) {
        throw std::runtime_error("Only file:// URIs are supported: " + uri);
    }

    const std::string encoded = uri.substr(7);
    std::string decoded;
    decoded.reserve(encoded.size());

    for (std::size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] == '%' && i + 2 < encoded.size()) {
            const int hi = hex_value(encoded[i + 1]);
            const int lo = hex_value(encoded[i + 2]);
            if (hi >= 0 && lo >= 0) {
                decoded.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        decoded.push_back(encoded[i]);
    }

    return std::filesystem::path(decoded).lexically_normal();
}

} // namespace clangllm
