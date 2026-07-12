cmake_minimum_required(VERSION 3.20)
project(clang_llm_client VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost 1.75 REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl)

add_executable(clang-llm
    src/main.cpp
    src/boost_json.cpp
    src/Uri.cpp
    src/SourceStore.cpp
    src/CompilationDatabase.cpp
    src/LspClient.cpp
    src/ClangdAnalyzer.cpp
    src/Prompt.cpp
    src/LlmClient.cpp
)

target_include_directories(clang-llm PRIVATE
    include
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(clang-llm PRIVATE
    Threads::Threads
    PkgConfig::LIBCURL
)

target_compile_options(clang-llm PRIVATE
    -Wall -Wextra -Wpedantic
)
