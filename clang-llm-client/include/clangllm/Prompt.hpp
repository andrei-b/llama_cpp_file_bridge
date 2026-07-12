#pragma once

#include "clangllm/Types.hpp"

#include <boost/json.hpp>

namespace clangllm {

boost::json::array build_review_messages(const FunctionContext& context);
boost::json::object context_to_json(const FunctionContext& context);

} // namespace clangllm
