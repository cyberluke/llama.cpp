// repro-21771.cpp
//
// Deterministic parser regression harness for llama.cpp issue #21771
// ("Qwen3 TAG_WITH_TAGGED tool format: p.json() fails on array<object>
//  parameter values; partial tool_call leaked to client poisons multi-turn
//  history").
//
// The harness builds the chat-template autoparser (TAG_WITH_TAGGED route) for a
// given template and feeds the exact historical model outputs from the issue
// through the generated PEG parser:
//   - the complete output, and
//   - every streaming prefix output.substr(0, n) for n in 1..size.
//
// For every prefix it records:
//   parse result type (SUCCESS / NEED_MORE_INPUT / FAIL)
//   result.end
//   number of tool calls, tool name, arguments string
//   whether the arguments string monotonically extends the previous prefix
//
// For the complete input it additionally requires:
//   parse succeeds, exactly one expected tool call exists,
//   json::parse(arguments) succeeds, arguments equal expected JSON structurally.
//
// Exit code (used by `git bisect run`):
//   0 = the scored cases (the original bug's JSON-correctness repros) pass
//   1 = at least one scored case reproduces the issue-21771 parser failure
//
// Scored cases: "1b" (array<object>, definition-order) and "2" (array<string>
// + boolean, lowercase true). Informational cases: "1" (exact issue bytes,
// limit emitted before the required query), "2u" (Python literal True),
// "neg" (negative / malformed values).
//
// Usage:
//   repro-21771 <template-file> <detail-log-file> <case-id> [<case-id> ...]
//     case-id: 1 | 1b | 2 | 2u | neg
//
// Compile with -DREPRO_OLD_API for revisions whose autoparser used
// build_parser(inputs) instead of build_parser(inputs, generation_prompt).

#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "json-schema-to-grammar.h"
#include "peg-parser.h"

#if __has_include("parsers.h")
#include "parsers.h"
#define REPRO_HAVE_QWEN3_CODER 1
#endif

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// JSON helpers (work with nlohmann::ordered_json and common_json alike)
// ----------------------------------------------------------------------------

static json j_obj(std::initializer_list<std::pair<const char *, json>> items) {
    json o = json::object();
    for (const auto & kv : items) {
        o[kv.first] = kv.second;
    }
    return o;
}

static json j_arr(std::initializer_list<json> items) {
    json a = json::array();
    for (const auto & v : items) {
        a.push_back(v);
    }
    return a;
}

// Structural JSON equality, ignoring object key order.
// Uses only the API shared by nlohmann::ordered_json and common_json.
static bool jseq(const json & a, const json & b) {
    if (a.is_object() != b.is_object() || a.is_array() != b.is_array()) {
        return false;
    }
    if (a.is_object()) {
        if (a.size() != b.size()) {
            return false;
        }
        for (auto it = a.begin(); it != a.end(); ++it) {
            if (!b.contains(it.key()) || !jseq(it.value(), b.at(it.key()))) {
                return false;
            }
        }
        return true;
    }
    if (a.is_array()) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (!jseq(a[i], b[i])) {
                return false;
            }
        }
        return true;
    }
    return a.dump() == b.dump();
}

// ----------------------------------------------------------------------------
// Tool definitions (exact schemas from issue #21771)
// ----------------------------------------------------------------------------

static json make_tools_firecrawl_search(bool all_optional = false) {
    json schema = j_obj({
        {"type", "object"},
        {"properties", j_obj({
            {"query", j_obj({{"type", "string"}})},
            {"limit", j_obj({{"type", "integer"}})},
            {"sources", j_obj({
                {"type", "array"},
                {"items", j_obj({
                    {"type", "object"},
                    {"properties", j_obj({
                        {"type", j_obj({{"type", "string"}, {"enum", j_arr({"web", "images", "news"})}})},
                    })},
                    {"required", j_arr({"type"})},
                })},
            })},
        })},
    });
    if (!all_optional) {
        schema["required"] = j_arr({"query"});
    }
    return j_arr({
        j_obj({
            {"type", "function"},
            {"function", j_obj({
                {"name", "firecrawl_search"},
                {"description", "Search the web with Firecrawl"},
                {"parameters", schema},
            })},
        }),
    });
}

static json make_tools_firecrawl_scrape(bool all_optional = false) {
    json schema = j_obj({
        {"type", "object"},
        {"properties", j_obj({
            {"url", j_obj({{"type", "string"}})},
            {"formats", j_obj({
                {"type", "array"},
                {"items", j_obj({{"type", "string"}})},
            })},
            {"onlyMainContent", j_obj({{"type", "boolean"}})},
        })},
    });
    if (!all_optional) {
        schema["required"] = j_arr({"url"});
    }
    return j_arr({
        j_obj({
            {"type", "function"},
            {"function", j_obj({
                {"name", "firecrawl_scrape"},
                {"description", "Scrape a URL with Firecrawl"},
                {"parameters", schema},
            })},
        }),
    });
}

// ----------------------------------------------------------------------------
// Historical model outputs
// ----------------------------------------------------------------------------

struct case_def {
    const char * id;       // 1, 1b, 1o, 2, 2o, 2ou, neg
    const char * tool;     // firecrawl_search | firecrawl_scrape
    const char * function; // function name
    bool        all_optional; // schema has no "required" array
    // parameter values in emission order
    std::vector<std::pair<std::string, std::string>> params;
    const char * expected_json; // expected arguments JSON for the valid cases
    bool scored;                // participates in the exit code
};

static const std::vector<case_def> CASES = {
    {
        "1", "firecrawl_search", "firecrawl_search", false,
        {
            {"limit",     "10"},
            {"query",     "pop culture trends USA March April 2026 movies music TV shows celebrities"},
            {"sources",   "[{\"type\": \"web\"}, {\"type\": \"news\"}]"},
        },
        "{\"limit\":10,\"query\":\"pop culture trends USA March April 2026 movies music TV shows celebrities\",\"sources\":[{\"type\":\"web\"},{\"type\":\"news\"}]}",
        false, // exact issue bytes: optional limit emitted before required query
    },
    {
        "1b", "firecrawl_search", "firecrawl_search", false,
        {
            {"query",     "pop culture trends USA March April 2026 movies music TV shows celebrities"},
            {"limit",     "10"},
            {"sources",   "[{\"type\": \"web\"}, {\"type\": \"news\"}]"},
        },
        "{\"limit\":10,\"query\":\"pop culture trends USA March April 2026 movies music TV shows celebrities\",\"sources\":[{\"type\":\"web\"},{\"type\":\"news\"}]}",
        true, // array<object> value parsing, definition order
    },
    {
        "1o", "firecrawl_search", "firecrawl_search", true,
        {
            {"limit",     "10"},
            {"query",     "pop culture trends USA March April 2026 movies music TV shows celebrities"},
            {"sources",   "[{\"type\": \"web\"}, {\"type\": \"news\"}]"},
        },
        "{\"limit\":10,\"query\":\"pop culture trends USA March April 2026 movies music TV shows celebrities\",\"sources\":[{\"type\":\"web\"},{\"type\":\"news\"}]}",
        true, // exact issue bytes with all-optional schema (issue's real failure mode)
    },
    {
        "2", "firecrawl_scrape", "firecrawl_scrape", false,
        {
            {"url",             "https://thehill.com/example"},
            {"formats",         "[\"markdown\"]"},
            {"onlyMainContent", "true"},
        },
        "{\"url\":\"https://thehill.com/example\",\"formats\":[\"markdown\"],\"onlyMainContent\":true}",
        true, // array<string> + boolean, lowercase true
    },
    {
        "2o", "firecrawl_scrape", "firecrawl_scrape", true,
        {
            {"url",             "https://thehill.com/example"},
            {"formats",         "[\"markdown\"]"},
            {"onlyMainContent", "true"},
        },
        "{\"url\":\"https://thehill.com/example\",\"formats\":[\"markdown\"],\"onlyMainContent\":true}",
        true, // exact issue bytes (comment 2), all-optional schema
    },
    {
        "2u", "firecrawl_scrape", "firecrawl_scrape", false,
        {
            {"url",             "https://thehill.com/example"},
            {"formats",         "[\"markdown\"]"},
            {"onlyMainContent", "True"}, // Python literal, not valid JSON
        },
        "{\"url\":\"https://thehill.com/example\",\"formats\":[\"markdown\"],\"onlyMainContent\":true}",
        false, // Python literal compatibility: informational (Phase G)
    },
    {
        "2ou", "firecrawl_scrape", "firecrawl_scrape", true,
        {
            {"url",             "https://thehill.com/example"},
            {"formats",         "[\"markdown\"]"},
            {"onlyMainContent", "True"}, // Python literal, not valid JSON
        },
        "{\"url\":\"https://thehill.com/example\",\"formats\":[\"markdown\"],\"onlyMainContent\":true}",
        false, // Python literal compatibility, all-optional: informational (Phase G)
    },
    {
        "neg", "firecrawl_scrape", "firecrawl_scrape", false,
        {
            {"url",             "https://thehill.com/example"},
            {"formats",         "[\"markdown\"]"},
            {"onlyMainContent", "ten"}, // invalid boolean spelling
        },
        nullptr,
        false, // negative test (Phase H): informational
    },
};

// ----------------------------------------------------------------------------
// Template loading
// ----------------------------------------------------------------------------

static std::string read_file(const std::string & path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        std::fprintf(stderr, "error: cannot open template file: %s\n", path.c_str());
        std::exit(2);
    }
    std::ostringstream buf;
    buf << fin.rdbuf();
    return buf.str();
}

// Build the model output for a case using the template's detected markers.
// The wrapper and parameter tags come from the analysis; the values are the
// exact bytes from the issue.
static std::string build_output(const std::string & /* tmpl_source */,
                                const autoparser::autoparser & analysis,
                                const case_def & c) {
    const auto & fmt  = analysis.tools.format;
    const auto & fn   = analysis.tools.function;
    const auto & args = analysis.tools.arguments;

    std::string opener = !fmt.section_start.empty() ? fmt.section_start : fmt.per_call_start;
    std::string closer = !fmt.section_end.empty() ? fmt.section_end : fmt.per_call_end;
    if (opener.empty()) {
        opener = "<tool_call>\n";
    }
    if (closer.empty()) {
        closer = "</tool_call>";
    }

    std::string out = opener;
    out += "<function=" + std::string(c.function) + fn.name_suffix;
    bool first = true;
    for (const auto & p : c.params) {
        if (!first) {
            out += "\n";
        }
        first = false;
        out += "<parameter=" + p.first + args.name_suffix + p.second + args.value_suffix;
    }
    out += fn.close;
    out += closer;
    return out;
}

// ----------------------------------------------------------------------------
// Per-prefix parse driver
// ----------------------------------------------------------------------------

struct prefix_record {
    size_t      n;        // prefix length
    const char *type;     // SUCCESS / NEED_MORE / FAIL
    size_t      end;      // result.end
    size_t      ncalls;
    std::string name;
    std::string args;
    bool        monotonic; // args extends previous prefix's args
};

// Debug mode: parse only prefix n with the DEBUG flag and return the raw result.
static prefix_record parse_prefix_debug(const common_peg_arena & parser, const std::string & input) {
    prefix_record rec;
    rec.n = input.size();
    rec.monotonic = true;

    common_peg_parse_context ctx(input, common_peg_parse_flags(COMMON_PEG_PARSE_FLAG_LENIENT | COMMON_PEG_PARSE_FLAG_DEBUG));
    auto result = parser.parse(ctx);
    switch (result.type) {
        case COMMON_PEG_PARSE_RESULT_SUCCESS:         rec.type = "SUCCESS";  break;
        case COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT: rec.type = "NEED_MORE"; break;
        default:                                      rec.type = "FAIL";     break;
    }
    rec.end = result.end;
    common_chat_msg msg;
    common_chat_peg_mapper mapper(msg);
    mapper.from_ast(ctx.ast, result);
    rec.ncalls = msg.tool_calls.size();
    if (!msg.tool_calls.empty()) {
        rec.name = msg.tool_calls[0].name;
        rec.args = msg.tool_calls[0].arguments;
    }
    return rec;
}

static prefix_record parse_prefix(const common_peg_arena & parser, const std::string & input,
                                  const std::string & prev_args) {
    prefix_record rec;
    rec.n = input.size();
    rec.monotonic = true;

    common_peg_parse_context ctx(input, COMMON_PEG_PARSE_FLAG_LENIENT);
    auto result = parser.parse(ctx);

    switch (result.type) {
        case COMMON_PEG_PARSE_RESULT_SUCCESS:        rec.type = "SUCCESS";       break;
        case COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT: rec.type = "NEED_MORE";     break;
        default:                                     rec.type = "FAIL";          break;
    }
    rec.end = result.end;

    common_chat_msg msg;
    common_chat_peg_mapper mapper(msg);
    mapper.from_ast(ctx.ast, result);

    rec.ncalls = msg.tool_calls.size();
    if (!msg.tool_calls.empty()) {
        rec.name = msg.tool_calls[0].name;
        rec.args = msg.tool_calls[0].arguments;
        if (!prev_args.empty() && rec.args.size() < prev_args.size()) {
            rec.monotonic = false;
        } else if (!prev_args.empty() && rec.args.compare(0, prev_args.size(), prev_args) != 0) {
            rec.monotonic = false;
        }
    }
    return rec;
}

// ----------------------------------------------------------------------------
// Case runner (autoparser route)
// ----------------------------------------------------------------------------

// Server mode: build the parser exactly like the server (peg_generator with the
// real generation prompt) and parse the effective input
// (generation_prompt + reasoning preamble + model output).
static const char * REASONING_PREAMBLE =
    "The user asked about pop culture trends in the USA for March and April 2026. "
    "I should use the firecrawl search tool to find current information about "
    "movies, music, TV shows and celebrities. I will set a limit of ten results "
    "and query for the relevant trends. The sources should include web and news "
    "results to cover both general articles and dedicated entertainment news.";

static int run_case_server_mode(const std::string & case_id, const std::string & tmpl_source,
                                const std::string & tmpl_path, std::ostream & log, bool old_api) {
    const case_def * cd = nullptr;
    for (const auto & c : CASES) {
        if (case_id == c.id) {
            cd = &c;
            break;
        }
    }
    if (!cd) {
        return 2;
    }

    common_chat_template tmpl(tmpl_source, "", "");

    autoparser::generation_params inputs;
    inputs.messages = j_arr({j_obj({{"role", "user"}, {"content", "hi"}})});
    inputs.tools = std::string(cd->tool) == "firecrawl_search" ? make_tools_firecrawl_search(cd->all_optional) : make_tools_firecrawl_scrape(cd->all_optional);
    inputs.reasoning_format    = COMMON_REASONING_FORMAT_NONE;
    inputs.parallel_tool_calls = false;
    inputs.add_generation_prompt = true; // match the server: parser input is prefixed with the generation prompt

    auto params = autoparser::peg_generator::generate_parser(tmpl, inputs);
    common_peg_arena arena;
    arena.load(params.parser);

    log << "=== server-mode case " << case_id << " on " << tmpl_path << " ===\n";
    log << "generation_prompt: [" << params.generation_prompt << "]\n";

    std::string body = "<tool_call>\n<function=" + std::string(cd->function) + ">\n";
    for (const auto & p : cd->params) {
        body += "<parameter=" + p.first + ">\n" + p.second + "\n</parameter>\n";
    }
    body += "</function>\n</tool_call>";

    std::string effective = params.generation_prompt + REASONING_PREAMBLE + "\n response\n\n" + body;
    log << "effective input (" << effective.size() << " bytes):\n" << effective << "\n";

    prefix_record full = parse_prefix(arena, effective, "");
    log << "complete parse: type=" << full.type << " end=" << full.end
        << " calls=" << full.ncalls;
    if (!full.name.empty()) {
        log << " name=" << full.name << " args=" << full.args;
    }
    log << "\n";

    bool complete_ok = false;
    bool args_valid  = false;
    bool args_equal  = false;
    if (std::string(full.type) == "SUCCESS" && full.ncalls == 1 && full.name == cd->function && cd->expected_json) {
        try {
            json actual   = json::parse(full.args);
            json expected = json::parse(cd->expected_json);
            args_valid = true;
            args_equal = jseq(actual, expected);
            complete_ok = args_equal;
        } catch (const std::exception & e) {
            log << "complete parse: arguments JSON error: " << e.what() << "\n";
        }
    }
    log << "complete checks: valid_json=" << (args_valid ? "yes" : "no")
        << " structurally_equal=" << (args_equal ? "yes" : "no") << "\n";

    size_t fail_count = 0;
    size_t need_count = 0;
    size_t succ_count = 0;
    size_t first_fail = 0;
    std::string prev_args;

    for (size_t n = 1; n <= effective.size(); ++n) {
        std::string prefix = effective.substr(0, n);
        prefix_record rec = parse_prefix(arena, prefix, prev_args);
        if (std::string(rec.type) == "FAIL") {
            ++fail_count;
            if (first_fail == 0) {
                first_fail = n;
            }
        } else if (std::string(rec.type) == "NEED_MORE") {
            ++need_count;
        } else {
            ++succ_count;
        }
        prev_args = rec.args;
    }
    log << "prefix summary: success=" << succ_count << " need_more=" << need_count
        << " fail=" << fail_count << " first_fail_at=" << (first_fail ? first_fail : 0) << "\n";

    bool ok = complete_ok && fail_count == 0;
    log << "case verdict: " << (ok ? "PASS" : "FAIL") << (cd->scored ? " (scored)" : " (informational)") << "\n";
    log << std::endl;

    return cd->scored ? (ok ? 0 : 1) : 0;
}

static int run_case(const std::string & case_id, const std::string & tmpl_source,
                    const std::string & tmpl_path, std::ostream & log,
                    bool old_api) {
    const case_def * cd = nullptr;
    for (const auto & c : CASES) {
        if (case_id == c.id) {
            cd = &c;
            break;
        }
    }
    if (!cd) {
        std::fprintf(stderr, "error: unknown case id: %s\n", case_id.c_str());
        return 2;
    }

    common_chat_template tmpl(tmpl_source, "", "");

    autoparser::autoparser analysis;
    analysis.analyze_template(tmpl);

    autoparser::generation_params inputs;
    inputs.messages = j_arr({j_obj({{"role", "user"}, {"content", "hi"}})});
    inputs.tools = std::string(cd->tool) == "firecrawl_search" ? make_tools_firecrawl_search(cd->all_optional) : make_tools_firecrawl_scrape(cd->all_optional);
    inputs.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
    inputs.parallel_tool_calls = false;

#ifdef REPRO_OLD_API
    auto parser = analysis.build_parser(inputs);
#else
    auto parser = analysis.build_parser(inputs, "");
#endif

    log << "=== case " << case_id << " on " << tmpl_path << " ===\n";
    log << "tool format mode:      " << int(analysis.tools.format.mode) << "\n";
    log << "section_start:         [" << analysis.tools.format.section_start << "]\n";
    log << "section_end:           [" << analysis.tools.format.section_end << "]\n";
    log << "per_call_start:        [" << analysis.tools.format.per_call_start << "]\n";
    log << "per_call_end:          [" << analysis.tools.format.per_call_end << "]\n";
    log << "name_prefix:           [" << analysis.tools.function.name_prefix << "]\n";
    log << "name_suffix:           [" << analysis.tools.function.name_suffix << "]\n";
    log << "function close:        [" << analysis.tools.function.close << "]\n";
    log << "arg name_prefix:       [" << analysis.tools.arguments.name_prefix << "]\n";
    log << "arg name_suffix:       [" << analysis.tools.arguments.name_suffix << "]\n";
    log << "arg value_prefix:      [" << analysis.tools.arguments.value_prefix << "]\n";
    log << "arg value_suffix:      [" << analysis.tools.arguments.value_suffix << "]\n";
    log << "arg separator:         [" << analysis.tools.arguments.separator << "]\n";
    log << "parameter order:       [";
    for (size_t i = 0; i < analysis.tools.format.parameter_order.size(); ++i) {
        log << (i ? "," : "") << analysis.tools.format.parameter_order[i];
    }
    log << "]\n";

#ifdef REPRO_DUMP_PARSER
    log << "--- parser dump ---\n" << parser.to_json().dump(2) << "\n--- end parser dump ---\n";
#endif

    std::string output = build_output(tmpl_source, analysis, *cd);
    log << "output (" << output.size() << " bytes):\n" << output << "\n";

    // ---- complete input ----
    prefix_record full = parse_prefix(parser, output, "");
    log << "complete parse: type=" << full.type << " end=" << full.end
        << " calls=" << full.ncalls;
    if (!full.name.empty()) {
        log << " name=" << full.name << " args=" << full.args;
    }
    log << "\n";

    bool complete_ok = false;
    bool args_valid  = false;
    bool args_equal  = false;
    if (std::string(full.type) == "SUCCESS" && full.ncalls == 1 && full.name == cd->function && cd->expected_json) {
        try {
            json actual   = json::parse(full.args);
            json expected = json::parse(cd->expected_json);
            args_valid = true;
            args_equal = jseq(actual, expected);
            complete_ok = args_equal;
        } catch (const std::exception & e) {
            log << "complete parse: arguments JSON error: " << e.what() << "\n";
        }
    }
    log << "complete checks: valid_json=" << (args_valid ? "yes" : "no")
        << " structurally_equal=" << (args_equal ? "yes" : "no") << "\n";

    // ---- every streaming prefix ----
    size_t fail_count = 0;
    size_t need_count = 0;
    size_t succ_count = 0;
    size_t first_fail = 0;
    std::string prev_args;

    for (size_t n = 1; n <= output.size(); ++n) {
        std::string prefix = output.substr(0, n);
        prefix_record rec = parse_prefix(parser, prefix, prev_args);
        log << "P " << rec.n << " " << rec.type << " end=" << rec.end
            << " calls=" << rec.ncalls;
        if (!rec.name.empty()) {
            log << " name=" << rec.name << " args=[" << rec.args << "]";
        }
        log << " mono=" << (rec.monotonic ? "yes" : "no") << "\n";

        if (std::string(rec.type) == "FAIL") {
            ++fail_count;
            if (first_fail == 0) {
                first_fail = n;
            }
        } else if (std::string(rec.type) == "NEED_MORE") {
            ++need_count;
        } else {
            ++succ_count;
        }
        prev_args = rec.args;
    }

    log << "prefix summary: success=" << succ_count << " need_more=" << need_count
        << " fail=" << fail_count << " first_fail_at=" << (first_fail ? first_fail : 0) << "\n";

    bool ok = complete_ok && fail_count == 0;
    log << "case verdict: " << (ok ? "PASS" : "FAIL") << (cd->scored ? " (scored)" : " (informational)") << "\n";
    log << std::endl;

    if (cd->scored) {
        return ok ? 0 : 1;
    }
    return 0;
}

#ifdef REPRO_HAVE_QWEN3_CODER

// ----------------------------------------------------------------------------
// Case runner (Qwen3-Coder specialized parser route, master only)
// ----------------------------------------------------------------------------

static int run_case_qwen3_coder(const std::string & case_id, const std::string & tmpl_source,
                                std::ostream & log) {
    const case_def * cd = nullptr;
    for (const auto & c : CASES) {
        if (case_id == c.id) {
            cd = &c;
            break;
        }
    }
    if (!cd) {
        return 2;
    }

    common_chat_template tmpl(tmpl_source, "", "");

    autoparser::generation_params inputs;
    inputs.messages = j_arr({j_obj({{"role", "user"}, {"content", "hi"}})});
    inputs.tools = std::string(cd->tool) == "firecrawl_search" ? make_tools_firecrawl_search(cd->all_optional) : make_tools_firecrawl_scrape(cd->all_optional);
    inputs.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
    inputs.parallel_tool_calls = false;

    auto params = common_chat_params_init_qwen3_coder(tmpl, inputs);

    // The qwen3-coder parser is stored serialized (parser.save()); load the arena.
    common_peg_arena arena;
    arena.load(params.parser);

    // The qwen3-coder parser expects the generation prefix <|im_start|>assistant\n.
    std::string gen  = params.generation_prompt;
    std::string body = "<tool_call>\n<function=" + std::string(cd->function) + ">\n";
    for (const auto & p : cd->params) {
        body += "<parameter=" + p.first + ">\n" + p.second + "\n</parameter>\n";
    }
    body += "</function>\n</tool_call>";
    std::string output = gen + body;

    log << "=== qwen3-coder case " << case_id << " ===\n";
    log << "output (" << output.size() << " bytes):\n" << output << "\n";

    prefix_record full = parse_prefix(arena, output, "");
    log << "complete parse: type=" << full.type << " end=" << full.end
        << " calls=" << full.ncalls;
    if (!full.name.empty()) {
        log << " name=" << full.name << " args=" << full.args;
    }
    log << "\n";

    bool complete_ok = false;
    bool args_valid  = false;
    bool args_equal  = false;
    if (std::string(full.type) == "SUCCESS" && full.ncalls == 1 && full.name == cd->function && cd->expected_json) {
        try {
            json actual   = json::parse(full.args);
            json expected = json::parse(cd->expected_json);
            args_valid = true;
            args_equal = jseq(actual, expected);
            complete_ok = args_equal;
        } catch (const std::exception & e) {
            log << "complete parse: arguments JSON error: " << e.what() << "\n";
        }
    }
    log << "complete checks: valid_json=" << (args_valid ? "yes" : "no")
        << " structurally_equal=" << (args_equal ? "yes" : "no") << "\n";

    size_t fail_count = 0;
    size_t need_count = 0;
    size_t succ_count = 0;
    size_t first_fail = 0;
    std::string prev_args;

    for (size_t n = 1; n <= output.size(); ++n) {
        std::string prefix = output.substr(0, n);
        prefix_record rec = parse_prefix(arena, prefix, prev_args);
        log << "P " << rec.n << " " << rec.type << " end=" << rec.end
            << " calls=" << rec.ncalls;
        if (!rec.name.empty()) {
            log << " name=" << rec.name << " args=[" << rec.args << "]";
        }
        log << " mono=" << (rec.monotonic ? "yes" : "no") << "\n";
        if (std::string(rec.type) == "FAIL") {
            ++fail_count;
            if (first_fail == 0) {
                first_fail = n;
            }
        } else if (std::string(rec.type) == "NEED_MORE") {
            ++need_count;
        } else {
            ++succ_count;
        }
        prev_args = rec.args;
    }

    log << "prefix summary: success=" << succ_count << " need_more=" << need_count
        << " fail=" << fail_count << " first_fail_at=" << (first_fail ? first_fail : 0) << "\n";

    bool ok = complete_ok && fail_count == 0;
    log << "case verdict: " << (ok ? "PASS" : "FAIL") << (cd->scored ? " (scored)" : " (informational)") << "\n";
    log << std::endl;

    return cd->scored ? (ok ? 0 : 1) : 0;
}

#endif // REPRO_HAVE_QWEN3_CODER

// ----------------------------------------------------------------------------

int main(int argc, char * argv[]) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <template-file> <detail-log-file> <case-id> [case-id ...]\n", argv[0]);
        return 2;
    }

    std::string tmpl_path = argv[1];
    std::string log_path  = argv[2];
    std::string tmpl_source = read_file(tmpl_path);

    std::ofstream log(log_path, std::ios::app);
    if (!log.is_open()) {
        std::fprintf(stderr, "error: cannot open log file: %s\n", log_path.c_str());
        return 2;
    }

    bool old_api = false;
#ifdef REPRO_OLD_API
    old_api = true;
#endif
    log << "## repro-21771 run: template=" << tmpl_path << " old_api=" << (old_api ? "yes" : "no") << "\n";

    bool is_qwen3_coder_template = tmpl_path.find("Qwen3-Coder") != std::string::npos;

    // Grammar dump mode: repro-21771 <template> <log> grammar:<case-id>
    std::string grammar_case;
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("grammar:", 0) == 0) {
            grammar_case = arg.substr(8);
        }
    }
    if (!grammar_case.empty()) {
        const case_def * cd = nullptr;
        for (const auto & c : CASES) {
            if (grammar_case == c.id) {
                cd = &c;
                break;
            }
        }
        if (!cd) {
            std::fprintf(stderr, "error: unknown grammar case\n");
            return 2;
        }
        common_chat_template tmpl(tmpl_source, "", "");
        autoparser::autoparser analysis;
        analysis.analyze_template(tmpl);
        autoparser::generation_params inputs;
        inputs.messages = j_arr({j_obj({{"role", "user"}, {"content", "hi"}})});
        inputs.tools = std::string(cd->tool) == "firecrawl_search" ? make_tools_firecrawl_search(cd->all_optional) : make_tools_firecrawl_scrape(cd->all_optional);
        inputs.reasoning_format    = COMMON_REASONING_FORMAT_NONE;
        inputs.parallel_tool_calls = false;
        inputs.tool_choice         = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
#ifdef REPRO_OLD_API
        auto g_parser = analysis.build_parser(inputs);
#else
        auto g_parser = analysis.build_parser(inputs, "");
#endif
        std::string grammar_str = build_grammar([&](const common_grammar_builder & builder) {
            g_parser.build_grammar(builder, /* lazy */ false);
        });
        std::printf("=== GRAMMAR for case %s (%zu bytes) ===\n%s\n=== END GRAMMAR ===\n", grammar_case.c_str(), grammar_str.size(), grammar_str.c_str());
        return 0;
    }

    // Optional debug mode: repro-21771 <template> <log> dbg:<case>:<prefix>
    std::string dbg_case;
    size_t      dbg_prefix = 0;
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("dbg:", 0) == 0) {
            size_t colon = arg.find(':', 4);
            if (colon != std::string::npos) {
                dbg_case   = arg.substr(4, colon - 4);
                dbg_prefix = (size_t) std::atoll(arg.substr(colon + 1).c_str());
            }
        }
    }
    if (!dbg_case.empty()) {
        const case_def * cd = nullptr;
        for (const auto & c : CASES) {
            if (dbg_case == c.id) {
                cd = &c;
                break;
            }
        }
        if (!cd) {
            std::fprintf(stderr, "error: unknown dbg case\n");
            return 2;
        }
        common_chat_template tmpl(tmpl_source, "", "");
        autoparser::autoparser analysis;
        analysis.analyze_template(tmpl);
        autoparser::generation_params inputs;
        inputs.messages = j_arr({j_obj({{"role", "user"}, {"content", "hi"}})});
        inputs.tools = std::string(cd->tool) == "firecrawl_search" ? make_tools_firecrawl_search(cd->all_optional) : make_tools_firecrawl_scrape(cd->all_optional);
        inputs.reasoning_format    = COMMON_REASONING_FORMAT_NONE;
        inputs.parallel_tool_calls = false;
#ifdef REPRO_OLD_API
        auto dbg_parser = analysis.build_parser(inputs);
#else
        auto dbg_parser = analysis.build_parser(inputs, "");
#endif
        std::string dbg_output = build_output(tmpl_source, analysis, *cd);
        if (dbg_prefix > dbg_output.size()) {
            dbg_prefix = dbg_output.size();
        }
        std::fprintf(stderr, "=== DEBUG parse of case %s prefix %zu ===\n", dbg_case.c_str(), dbg_prefix);
        std::fprintf(stderr, "--- input ---\n%s\n--- end input ---\n", dbg_output.substr(0, dbg_prefix).c_str());
        prefix_record rec = parse_prefix_debug(dbg_parser, dbg_output.substr(0, dbg_prefix));
        std::fprintf(stderr, "result: type=%s end=%zu calls=%zu name=%s args=[%s]\n",
            rec.type, rec.end, rec.ncalls, rec.name.c_str(), rec.args.c_str());
        return 0;
    }

    int exit_code = 0;
    for (int i = 3; i < argc; ++i) {
        std::string case_id = argv[i];

        // Server-mode variants: sv:<case-id> build the parser like the server
        // (peg_generator with the real generation prompt) and parse the
        // effective input including a reasoning preamble.
        bool server_mode = case_id.rfind("sv:", 0) == 0;
        std::string base_case = server_mode ? case_id.substr(3) : case_id;

        // Informational cases never affect the exit code.
        bool scored = base_case == "1b" || base_case == "1o" || base_case == "2" || base_case == "2o";

        int rc = server_mode
            ? run_case_server_mode(base_case, tmpl_source, tmpl_path, log, old_api)
            : run_case(base_case, tmpl_source, tmpl_path, log, old_api);
        if (scored && rc != 0) {
            exit_code = 1;
        }

#ifdef REPRO_HAVE_QWEN3_CODER
        // Qwen3-Coder specialized parser route (only where the API exists).
        if (is_qwen3_coder_template) {
            int rcq = run_case_qwen3_coder(case_id, tmpl_source, log);
            if (scored && rcq != 0) {
                exit_code = 1;
            }
        }
#endif
    }

    log << "## repro-21771 exit code: " << exit_code << "\n";
    log.close();
    std::printf("repro-21771 exit code: %d\n", exit_code);
    return exit_code;
}
