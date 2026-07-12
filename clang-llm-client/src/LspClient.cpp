#include "clangllm/LspClient.hpp"
#include "clangllm/Uri.hpp"

#include <boost/json.hpp>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <system_error>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace clangllm {
namespace json = boost::json;

namespace {

void write_all(int fd, const char* data, std::size_t size) {
    while (size > 0) {
        const auto written = ::write(fd, data, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::system_error(errno, std::generic_category(), "write to clangd");
        }
        data += written;
        size -= static_cast<std::size_t>(written);
    }
}

bool read_exact(int fd, char* data, std::size_t size) {
    while (size > 0) {
        const auto count = ::read(fd, data, size);
        if (count == 0) {
            return false;
        }
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::system_error(errno, std::generic_category(), "read from clangd");
        }
        data += count;
        size -= static_cast<std::size_t>(count);
    }
    return true;
}

bool read_line(int fd, std::string& line) {
    line.clear();
    char c = '\0';
    while (true) {
        const auto count = ::read(fd, &c, 1);
        if (count == 0) {
            return !line.empty();
        }
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::system_error(errno, std::generic_category(), "read clangd header");
        }
        if (c == '\n') {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            return true;
        }
        line.push_back(c);
    }
}

std::string progress_token_key(const json::value& token) {
    return json::serialize(token);
}

} // namespace

LspClient::LspClient(std::filesystem::path project_root,
                     std::filesystem::path build_directory,
                     std::string clangd_executable)
    : project_root_(std::filesystem::absolute(std::move(project_root)).lexically_normal()),
      build_directory_(std::filesystem::absolute(std::move(build_directory)).lexically_normal()),
      clangd_executable_(std::move(clangd_executable)) {
    std::signal(SIGPIPE, SIG_IGN);
}

LspClient::~LspClient() {
    stop();
}

void LspClient::start() {
    if (running_) {
        return;
    }

    int parent_to_child[2]{};
    int child_to_parent[2]{};
    if (::pipe(parent_to_child) != 0 || ::pipe(child_to_parent) != 0) {
        throw std::system_error(errno, std::generic_category(), "pipe");
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        throw std::system_error(errno, std::generic_category(), "fork");
    }

    if (pid == 0) {
        ::dup2(parent_to_child[0], STDIN_FILENO);
        ::dup2(child_to_parent[1], STDOUT_FILENO);
        ::close(parent_to_child[0]);
        ::close(parent_to_child[1]);
        ::close(child_to_parent[0]);
        ::close(child_to_parent[1]);

        const std::string compile_arg = "--compile-commands-dir=" + build_directory_.string();
        const char* argv[] = {
            clangd_executable_.c_str(),
            compile_arg.c_str(),
            "--background-index",
            "--clang-tidy",
            "--log=error",
            nullptr,
        };
        ::execvp(argv[0], const_cast<char* const*>(argv));
        std::cerr << "Failed to exec clangd: " << std::strerror(errno) << '\n';
        _exit(127);
    }

    child_pid_ = static_cast<int>(pid);
    child_stdin_ = parent_to_child[1];
    child_stdout_ = child_to_parent[0];
    ::close(parent_to_child[0]);
    ::close(child_to_parent[1]);

    running_ = true;
    reader_thread_ = std::thread([this] { reader_loop(); });
}

void LspClient::initialize() {
    json::object capabilities;
    capabilities["general"] = json::object{{"positionEncodings", json::array{"utf-8", "utf-16"}}};
    capabilities["offsetEncoding"] = json::array{"utf-8", "utf-16"};
    capabilities["window"] = json::object{{"workDoneProgress", true}};
    capabilities["textDocument"] = json::object{
        {"documentSymbol", json::object{{"hierarchicalDocumentSymbolSupport", true}}},
        {"references", json::object{{"container", true}}},
        {"publishDiagnostics", json::object{{"codeActionsInline", true}, {"categorySupport", true}}},
    };

    json::object params;
    params["processId"] = static_cast<std::int64_t>(::getpid());
    params["rootUri"] = path_to_uri(project_root_);
    params["capabilities"] = std::move(capabilities);
    params["initializationOptions"] = json::object{
        {"compilationDatabasePath", build_directory_.string()},
        {"clangdFileStatus", true},
    };

    (void)request("initialize", std::move(params));
    notify("initialized", json::object{});
}

json::value LspClient::request(const std::string& method, json::value params,
                               std::chrono::seconds timeout) {
    if (!running_) {
        throw std::runtime_error("clangd is not running");
    }

    const auto id = next_id_.fetch_add(1);
    json::object message;
    message["jsonrpc"] = "2.0";
    message["id"] = id;
    message["method"] = method;
    message["params"] = std::move(params);
    write_message(message);

    std::unique_lock lock(state_mutex_);
    const bool ready = state_cv_.wait_for(lock, timeout, [&] {
        return responses_.contains(id) || !running_ || !reader_error_.empty();
    });

    if (!ready) {
        throw std::runtime_error("Timed out waiting for clangd method: " + method);
    }
    if (!reader_error_.empty()) {
        throw std::runtime_error("clangd reader failed: " + reader_error_);
    }

    auto it = responses_.find(id);
    if (it == responses_.end()) {
        throw std::runtime_error("clangd stopped while handling: " + method);
    }

    json::value response = std::move(it->second);
    responses_.erase(it);
    lock.unlock();

    if (!response.is_object()) {
        throw std::runtime_error("Invalid clangd response for: " + method);
    }
    const auto& object = response.as_object();
    if (const auto* error = object.if_contains("error")) {
        throw std::runtime_error("clangd error for " + method + ": " + json::serialize(*error));
    }
    if (const auto* result = object.if_contains("result")) {
        return *result;
    }
    return nullptr;
}

void LspClient::notify(const std::string& method, json::value params) {
    if (!running_) {
        return;
    }
    json::object message;
    message["jsonrpc"] = "2.0";
    message["method"] = method;
    message["params"] = std::move(params);
    write_message(message);
}

void LspClient::write_message(const json::value& message) {
    const std::string body = json::serialize(message);
    const std::string header = "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n";
    std::scoped_lock lock(write_mutex_);
    write_all(child_stdin_, header.data(), header.size());
    write_all(child_stdin_, body.data(), body.size());
}

bool LspClient::read_frame(std::string& body) {
    std::size_t content_length = 0;
    std::string line;

    while (true) {
        if (!read_line(child_stdout_, line)) {
            return false;
        }
        if (line.empty()) {
            break;
        }
        constexpr std::string_view prefix = "Content-Length:";
        if (line.starts_with(prefix)) {
            content_length = static_cast<std::size_t>(std::stoull(line.substr(prefix.size())));
        }
    }

    if (content_length == 0) {
        throw std::runtime_error("clangd frame has no Content-Length");
    }

    body.resize(content_length);
    return read_exact(child_stdout_, body.data(), content_length);
}

void LspClient::reader_loop() {
    try {
        std::string body;
        while (running_ && read_frame(body)) {
            handle_message(json::parse(body));
        }
    } catch (const std::exception& error) {
        std::scoped_lock lock(state_mutex_);
        reader_error_ = error.what();
    }

    running_ = false;
    state_cv_.notify_all();
}

void LspClient::handle_message(json::value message) {
    if (!message.is_object()) {
        return;
    }
    const auto& object = message.as_object();
    const auto* method_value = object.if_contains("method");
    const auto* id_value = object.if_contains("id");

    if (method_value != nullptr && method_value->is_string()) {
        const std::string method(method_value->as_string().c_str());
        if (id_value != nullptr) {
            respond_to_server_request(object);
            return;
        }

        const auto* params_value = object.if_contains("params");
        if (method == "textDocument/publishDiagnostics" && params_value != nullptr &&
            params_value->is_object()) {
            const auto& params = params_value->as_object();
            const auto* uri = params.if_contains("uri");
            const auto* diagnostics = params.if_contains("diagnostics");
            if (uri != nullptr && uri->is_string() && diagnostics != nullptr && diagnostics->is_array()) {
                std::scoped_lock lock(state_mutex_);
                diagnostics_[std::string(uri->as_string().c_str())] = diagnostics->as_array();
            }
        } else if (method == "$/progress" && params_value != nullptr && params_value->is_object()) {
            const auto& params = params_value->as_object();
            const auto* token = params.if_contains("token");
            const auto* value = params.if_contains("value");
            if (token != nullptr && value != nullptr && value->is_object()) {
                const auto* kind = value->as_object().if_contains("kind");
                if (kind != nullptr && kind->is_string()) {
                    std::scoped_lock lock(state_mutex_);
                    progress_seen_ = true;
                    const std::string key = progress_token_key(*token);
                    const std::string kind_text(kind->as_string().c_str());
                    if (kind_text == "begin") {
                        active_progress_tokens_.insert(key);
                    } else if (kind_text == "end") {
                        active_progress_tokens_.erase(key);
                    }
                    state_cv_.notify_all();
                }
            }
        }
        return;
    }

    if (id_value != nullptr && id_value->is_int64()) {
        std::scoped_lock lock(state_mutex_);
        responses_[id_value->as_int64()] = std::move(message);
        state_cv_.notify_all();
    }
}

void LspClient::respond_to_server_request(const json::object& request_object) {
    json::object response;
    response["jsonrpc"] = "2.0";
    response["id"] = request_object.at("id");
    response["result"] = nullptr;
    write_message(response);
}

bool LspClient::wait_for_background_index(std::chrono::seconds timeout) {
    std::unique_lock lock(state_mutex_);
    return state_cv_.wait_for(lock, timeout, [&] {
        return progress_seen_ && active_progress_tokens_.empty();
    });
}

std::vector<std::string> LspClient::diagnostics_for_uri(const std::string& uri) const {
    std::vector<std::string> result;
    std::scoped_lock lock(state_mutex_);
    const auto it = diagnostics_.find(uri);
    if (it == diagnostics_.end()) {
        return result;
    }

    for (const auto& item : it->second) {
        if (!item.is_object()) {
            continue;
        }
        const auto& object = item.as_object();
        const auto* message = object.if_contains("message");
        if (message == nullptr || !message->is_string()) {
            continue;
        }
        std::string text(message->as_string().c_str());
        if (const auto* code = object.if_contains("code")) {
            text += " [" + json::serialize(*code) + "]";
        }
        result.push_back(std::move(text));
    }
    return result;
}

void LspClient::stop() {
    if (child_pid_ <= 0) {
        return;
    }

    if (running_) {
        try {
            (void)request("shutdown", json::object{}, std::chrono::seconds{5});
            notify("exit", json::object{});
        } catch (...) {
            // Fall through to process termination.
        }
    }

    force_stop();
}

void LspClient::force_stop() noexcept {
    if (child_stdin_ >= 0) {
        ::close(child_stdin_);
        child_stdin_ = -1;
    }

    if (child_pid_ > 0) {
        int status = 0;
        pid_t result = ::waitpid(child_pid_, &status, WNOHANG);
        if (result == 0) {
            ::kill(child_pid_, SIGTERM);
            (void)::waitpid(child_pid_, &status, 0);
        }
        child_pid_ = -1;
    }

    if (child_stdout_ >= 0) {
        ::close(child_stdout_);
        child_stdout_ = -1;
    }

    running_ = false;
    state_cv_.notify_all();
    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }
}

} // namespace clangllm
