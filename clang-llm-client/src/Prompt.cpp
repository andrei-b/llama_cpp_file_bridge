#include "clangllm/Prompt.hpp"

#include <boost/json.hpp>

namespace clangllm {
namespace json = boost::json;

json::object context_to_json(const FunctionContext& context) {
    json::array callers;
    for (const auto& caller : context.callers) {
        callers.push_back(json::object{
            {"caller", caller.caller_name},
            {"file", caller.file.string()},
            {"reference_line", caller.reference_line},
            {"reference_excerpt", caller.reference_excerpt},
            {"caller_implementation", caller.caller_implementation},
        });
    }

    json::array diagnostics;
    for (const auto& diagnostic : context.diagnostics) {
        diagnostics.push_back(json::value(diagnostic));
    }

    return json::object{
        {"function", context.qualified_name},
        {"usr", context.usr},
        {"file", context.symbol.file.string()},
        {"start_line", context.symbol.range.start.line + 1},
        {"hover_and_documentation", context.hover},
        {"declaration_excerpt", context.declaration_excerpt},
        {"implementation", context.implementation},
        {"reference_contexts", std::move(callers)},
        {"clang_diagnostics", std::move(diagnostics)},
    };
}

json::array build_review_messages(const FunctionContext& context) {
    static constexpr const char* system_prompt = R"PROMPT(
You are a conservative senior C++ semantic reviewer.

Your task is to compare a function's implementation with evidence about its intended contract. Evidence can come from its declaration, comments, types, caller assumptions, tests, assertions, and compiler/clang diagnostics.

Rules:
1. Call-site context is evidence, not a complete specification.
2. Do not invent requirements that are not supported by the supplied context.
3. Report only concrete, plausible correctness defects. Ignore style and subjective design preferences.
4. Pay special attention to nullability, ownership/lifetime, state transitions, bounds, units, error handling, concurrency, integer conversions, exception guarantees, and mismatched caller assumptions.
5. A reference context may be a declaration, address-taking expression, or callback registration rather than a direct call. Do not assume it is a call unless the code shows that.
6. If evidence is insufficient, put the concern in uncertainties rather than findings.
7. Return valid JSON only. Do not use Markdown fences.

Return this exact top-level shape:
{
  "observed_behavior": "brief factual summary",
  "inferred_contract": ["contract statement grounded in evidence"],
  "findings": [
    {
      "severity": "error|warning|note",
      "confidence": 0.0,
      "category": "short category",
      "summary": "one-sentence defect",
      "evidence": ["specific implementation/caller evidence"],
      "suggested_test": "a test that would confirm or reject the defect"
    }
  ],
  "uncertainties": ["missing information that prevents a conclusion"]
}
)PROMPT";

    return json::array{
        json::object{{"role", "system"}, {"content", system_prompt}},
        json::object{{"role", "user"},
                     {"content", "Review this C++ function context:\n" +
                                     json::serialize(context_to_json(context))}},
    };
}

} // namespace clangllm
