#include "clangllm/LlmClient.hpp"

#include <boost/json.hpp>
#include <curl/curl.h>

#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>

namespace clangllm {
namespace json = boost::json;

namespace {

std::size_t append_response(char* data, std::size_t size, std::size_t count, void* user_data) {
    const std::size_t bytes = size * count;
    static_cast<std::string*>(user_data)->append(data, bytes);
    return bytes;
}

struct CurlHandleDeleter {
    void operator()(CURL* handle) const noexcept {
        if (handle != nullptr) curl_easy_cleanup(handle);
    }
};

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.pop_back();
    }
    return value;
}

} // namespace

LlmClient::LlmClient(std::string endpoint, std::string model,
                     std::optional<std::string> api_key)
    : endpoint_(std::move(endpoint)), model_(std::move(model)), api_key_(std::move(api_key)) {
    static const int initialized = [] {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        return 1;
    }();
    (void)initialized;
}

json::value LlmClient::chat(const json::array& messages, double temperature,
                            int max_tokens) const {
    json::object payload{
        {"model", model_},
        {"messages", messages},
        {"temperature", temperature},
        {"max_tokens", max_tokens},
        {"stream", false},
    };
    const std::string request_body = json::serialize(payload);

    std::unique_ptr<CURL, CurlHandleDeleter> curl(curl_easy_init());
    if (!curl) {
        throw std::runtime_error("curl_easy_init failed");
    }

    curl_slist* raw_headers = nullptr;
    raw_headers = curl_slist_append(raw_headers, "Content-Type: application/json");
    if (api_key_ && !api_key_->empty()) {
        raw_headers = curl_slist_append(
            raw_headers, ("Authorization: Bearer " + *api_key_).c_str());
    }
    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)>
        headers(raw_headers, &curl_slist_free_all);

    std::string response_body;
    curl_easy_setopt(curl.get(), CURLOPT_URL, endpoint_.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get());
    curl_easy_setopt(curl.get(), CURLOPT_POST, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS, request_body.data());
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE,
                     static_cast<long>(request_body.size()));
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, append_response);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 300L);

    const CURLcode code = curl_easy_perform(curl.get());
    if (code != CURLE_OK) {
        throw std::runtime_error(std::string("LLM HTTP request failed: ") +
                                 curl_easy_strerror(code));
    }

    long status = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
    if (status < 200 || status >= 300) {
        throw std::runtime_error("LLM endpoint returned HTTP " +
                                 std::to_string(status) + ": " + response_body);
    }

    const auto response = json::parse(response_body);
    if (!response.is_object()) {
        throw std::runtime_error("LLM endpoint returned a non-object response");
    }
    const auto* choices = response.as_object().if_contains("choices");
    if (choices == nullptr || !choices->is_array() || choices->as_array().empty()) {
        throw std::runtime_error("LLM response contains no choices: " + response_body);
    }
    const auto& first = choices->as_array().front().as_object();
    const auto* message = first.if_contains("message");
    if (message == nullptr || !message->is_object()) {
        throw std::runtime_error("LLM response has no message: " + response_body);
    }
    const auto* content = message->as_object().if_contains("content");
    if (content == nullptr || !content->is_string()) {
        throw std::runtime_error("LLM response has no textual content: " + response_body);
    }

    return parse_model_json(content->as_string().c_str());
}

json::value parse_model_json(const std::string& input) {
    std::string text = trim(input);
    if (text.starts_with("```")) {
        const auto first_newline = text.find('\n');
        const auto last_fence = text.rfind("```");
        if (first_newline != std::string::npos && last_fence != std::string::npos &&
            last_fence > first_newline) {
            text = text.substr(first_newline + 1, last_fence - first_newline - 1);
        }
    }

    try {
        return json::parse(trim(text));
    } catch (const std::exception&) {
        return json::object{{"parse_error", true}, {"raw_model_output", input}};
    }
}

} // namespace clangllm
