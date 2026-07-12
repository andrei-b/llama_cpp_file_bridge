#pragma once

#include <boost/json.hpp>

#include <optional>
#include <string>

namespace clangllm {

class LlmClient {
public:
    LlmClient(std::string endpoint, std::string model,
              std::optional<std::string> api_key = std::nullopt);

    boost::json::value chat(const boost::json::array& messages,
                            double temperature = 0.1,
                            int max_tokens = 1400) const;

private:
    std::string endpoint_;
    std::string model_;
    std::optional<std::string> api_key_;
};

boost::json::value parse_model_json(const std::string& text);

} // namespace clangllm
