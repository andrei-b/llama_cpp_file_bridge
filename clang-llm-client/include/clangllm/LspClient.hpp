#pragma once

#include <boost/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace clangllm {

class LspClient {
public:
    LspClient(std::filesystem::path project_root,
              std::filesystem::path build_directory,
              std::string clangd_executable = "clangd");
    ~LspClient();

    LspClient(const LspClient&) = delete;
    LspClient& operator=(const LspClient&) = delete;

    void start();
    void initialize();
    void stop();

    boost::json::value request(const std::string& method,
                               boost::json::value params,
                               std::chrono::seconds timeout = std::chrono::seconds{60});
    void notify(const std::string& method, boost::json::value params);

    bool wait_for_background_index(std::chrono::seconds timeout);
    std::vector<std::string> diagnostics_for_uri(const std::string& uri) const;

private:
    void reader_loop();
    bool read_frame(std::string& body);
    void write_message(const boost::json::value& message);
    void handle_message(boost::json::value message);
    void respond_to_server_request(const boost::json::object& request);
    void force_stop() noexcept;

    std::filesystem::path project_root_;
    std::filesystem::path build_directory_;
    std::string clangd_executable_;

    int child_stdin_ = -1;
    int child_stdout_ = -1;
    int child_pid_ = -1;
    std::thread reader_thread_;
    std::atomic<bool> running_{false};
    std::atomic<std::int64_t> next_id_{1};

    mutable std::mutex state_mutex_;
    std::mutex write_mutex_;
    std::condition_variable state_cv_;
    std::unordered_map<std::int64_t, boost::json::value> responses_;
    std::unordered_map<std::string, boost::json::array> diagnostics_;
    std::set<std::string> active_progress_tokens_;
    bool progress_seen_ = false;
    std::string reader_error_;
};

} // namespace clangllm
