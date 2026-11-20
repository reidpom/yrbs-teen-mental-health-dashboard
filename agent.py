"""The "Ask the data" assistant: Claude with read-only tools over the project's precomputed results.
No Streamlit code here, so the tools can be tested on their own (python3 agent.py).
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
TABLE_INDEX = json.loads((DATA / "tables" / "index.json").read_text())
QUESTIONS = json.loads((DATA / "questions.json").read_text())
REPORT = (DATA / "report.md").read_text()
_students = pd.read_csv(DATA / "students.csv")
_qstats = pd.read_csv(DATA / "question_stats.csv")

DEFAULT_MODEL = "claude-opus-5-5"
# $ per million tokens: input, output, cache write (5 min), cache read
PRICES = {"claude-opus-5-5": (4.0, 20.0, 5.0, 0.20), "claude-sonnet-5-5": (2.0, 10.0, 2.5, 0.10), "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01)}
MAX_TOOL_ROUNDS = 6

SYSTEM_RULES = """You are the data assistant on a dashboard for a university group project (MIS 545, machine learning) about teen mental health, using the CDC Youth Risk Behavior Survey. Models were built on the 2023 survey (15,075 usable students) and tested once on the 2025 survey (9,686). The outcome is Q26: felt sad or hopeless almost every day for two weeks or more in the past year. The main exposure is Q80, social media use.

Your readers are classmates and instructors, many without a machine learning background.

How to answer:
- Answer only from the project's results: the tools below and the project write-up included after these rules. Get exact numbers from the tools when you can, and say which table or section they come from. If the results do not cover a question, say so plainly; do not fill gaps from general knowledge or other studies.
- Each student answered the survey once, so describe results as "linked to", "goes with" or "predicts". Never say social media or anything else "causes" sadness, even if asked to. Explain why if someone pushes.
- Social media figures compare the heaviest users ("more than once an hour", 32% of students) with students who do not use social media.
- Explain any technical term the first time in everyday words. Keep answers short: a direct answer first, then a few bullets. Round numbers sensibly (percentages to whole numbers, odds and scores to two or three decimals).
- Results describe groups of students. Do not speculate about any individual. You are not a clinician: if someone describes being in crisis, tell them to call or text 988 (Suicide & Crisis Lifeline) or talk to a trusted adult, and keep it brief and kind.
- Group rates with fewer than 30 students are unreliable; say so when you use them."""


# ----------------------------------------------------------------- tools
def list_tables() -> str:
    out = []
    for name, desc in TABLE_INDEX.items():
        cols = "" if name.endswith(".txt") else ", ".join(pd.read_csv(DATA / "tables" / name, nrows=0).columns)
        out.append(f"- {name}: {desc}" + (f" Columns: {cols}" if cols else ""))
    return "\n".join(out)


def read_table(name: str) -> str:
    if name not in TABLE_INDEX:
        return f"Unknown table '{name}'. Call list_tables to see the names."
    p = DATA / "tables" / name
    if name.endswith(".txt"):
        return p.read_text()[:8000]
    df = pd.read_csv(p)
    txt = df.round(4).to_csv(index=False)
    return txt if len(txt) < 12000 else txt[:12000] + f"\n... (cut; {len(df)} rows in all)"


def search_questions(keyword: str) -> str:
    k = keyword.lower().strip()
    hits = [(q, v) for q, v in QUESTIONS.items() if k in v["name"].lower() or k in v["label"].lower() or k == q.lower()
            or any(k in a.lower() for a in v["answers"].values())]
    if not hits:
        return f"No survey question matches '{keyword}'. Try a shorter keyword (e.g. 'sleep', 'bully', 'vape')."
    return "\n".join(f"{q} ({v['name']}; CDC label: {v['label']}). Answers: " + "; ".join(f"{c}={t}" for c, t in v["answers"].items()) for q, v in hits[:12])


def question_breakdown(question: str) -> str:
    q = question.upper().strip()
    sub = _qstats[_qstats.question == q]
    if sub.empty:
        return f"No data for '{question}'. Use search_questions to find the question code (like Q80)."
    sub = sub.assign(share=(sub.share * 100).round(1), sad_rate=(sub.sad_rate * 100).round(1))
    head = f"{q}: {QUESTIONS[q]['name']}. Weighted % of students giving each answer, and weighted % of those students who felt sad or hopeless (Q26)."
    return head + "\n" + sub[["year", "answer", "n", "share", "sad_rate"]].rename(columns={"share": "pct_of_students", "sad_rate": "pct_sad"}).to_csv(index=False)


GROUPS = {"sex": "sex", "grade": "grade", "social_media": "social", "kmeans_profile": "kmeans", "hierarchical_group": "ward",
          "risk_only_group": "risk_group", "year": "year"}


def group_sad_rate(group_by: str, year: str = "both", sex: str = "all") -> str:
    s = _students
    if year in ("2023", "2025"):
        s = s[s.year == int(year)]
    if sex in ("Girl", "Boy"):
        s = s[s.sex == sex]
    col = GROUPS[group_by]
    rows = [{"group": g, "students": len(x), "pct_sad_weighted": round(100 * np.average(x.sad, weights=x.weight), 1)}
            for g, x in s.dropna(subset=[col]).groupby(col)]
    note = "Groups for k-means / hierarchical / risk-only are numbered 1 (least sad) to 5 (most sad)."
    return f"Felt sad or hopeless by {group_by} (year={year}, sex={sex}). {note}\n" + pd.DataFrame(rows).to_csv(index=False)


TOOLS = [
    {"name": "list_tables", "description": "List the project's result tables with a description and column names for each. Call this first if unsure which table holds an answer.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "read_table", "description": "Return one result table as CSV (model scores, social media odds ratios by stage, SHAP importance, fairness, profiles, protective factors, data checks).",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "enum": list(TABLE_INDEX)}}, "required": ["name"], "additionalProperties": False}},
    {"name": "search_questions", "description": "Find survey questions by keyword; returns question codes (2023 numbering), plain names and answer options.",
     "input_schema": {"type": "object", "properties": {"keyword": {"type": "string", "description": "A word such as sleep, bully, vape, social media"}}, "required": ["keyword"], "additionalProperties": False}},
    {"name": "question_breakdown", "description": "For one survey question, the weighted share of students giving each answer and the weighted share of them who felt sad or hopeless, for 2023 and 2025.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string", "description": "Question code like Q80"}}, "required": ["question"], "additionalProperties": False}},
    {"name": "group_sad_rate", "description": "Weighted share of students who felt sad or hopeless, split by one grouping, optionally for one year and one sex.",
     "input_schema": {"type": "object", "properties": {
         "group_by": {"type": "string", "enum": list(GROUPS)},
         "year": {"type": "string", "enum": ["2023", "2025", "both"]},
         "sex": {"type": "string", "enum": ["Girl", "Boy", "all"]}}, "required": ["group_by"], "additionalProperties": False}},
]
for t in TOOLS:
    t["eager_input_streaming"] = True          # inputs stream as generated; validated below before any tool runs
FUNCS = {"list_tables": list_tables, "read_table": read_table, "search_questions": search_questions,
         "question_breakdown": question_breakdown, "group_sad_rate": group_sad_rate}
SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}


def validate(name: str, args) -> str | None:
    """Check a tool input against its schema; returns an error message or None."""
    if name not in SCHEMAS:
        return f"Unknown tool {name}"
    if not isinstance(args, dict):
        return "Tool input must be a JSON object"
    sch = SCHEMAS[name]
    for k in sch.get("required", []):
        if k not in args:
            return f"Missing required field '{k}'"
    for k, v in args.items():
        spec = sch["properties"].get(k)
        if spec is None:
            return f"Unexpected field '{k}'"
        if not isinstance(v, str) or not v.strip():
            return f"Field '{k}' must be a non-empty string"
        if "enum" in spec and v not in spec["enum"]:
            return f"Field '{k}' must be one of {spec['enum']}"
    return None


def run_tool(name: str, args) -> tuple[str, bool]:
    err = validate(name, args)
    if err:
        return f"Invalid input: {err}. Raw input: {json.dumps(args)[:500]}", True
    try:
        return FUNCS[name](**args), False
    except Exception as e:  # a tool bug should not crash the chat
        return f"Tool error: {type(e).__name__}: {e}", True


def system_blocks():
    """Frozen system prompt; the cache breakpoint sits after the report so tools + rules + report are cached together."""
    return [{"type": "text", "text": SYSTEM_RULES},
            {"type": "text", "text": "PROJECT WRITE-UP (source of truth for claims and wording):\n\n" + REPORT, "cache_control": {"type": "ephemeral"}}]


def cost(model: str, usage) -> float:
    p = PRICES.get(model)
    if not p or usage is None:
        return 0.0
    return (usage.input_tokens * p[0] + usage.output_tokens * p[1] + (usage.cache_creation_input_tokens or 0) * p[2]
            + (usage.cache_read_input_tokens or 0) * p[3]) / 1e6


def answer(client, model: str, messages: list, effort: str = "low"):
    """Runs one user turn. Yields ("text", chunk), ("tool", description) and finally ("done", info).
    `messages` is extended in place with the assistant turns and tool results (append-only)."""
    import anthropic
    spent, json_retries = 0.0, 0
    for _ in range(MAX_TOOL_ROUNDS + 1):
        try:
            with client.beta.messages.stream(
                model=model, max_tokens=8000, system=system_blocks(), tools=TOOLS, messages=messages,
                output_config={"effort": effort},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            ) as stream:
                for event in stream:
                    if event.type == "text":
                        yield "text", event.text
                final = stream.get_final_message()
        except ValueError:                          # tool input that could not be parsed at all: retry the turn
            json_retries += 1
            if json_retries > 2:
                yield "done", {"error": "The assistant produced a malformed request twice. Please rephrase.", "cost": spent}
                return
            continue
        except anthropic.AuthenticationError:
            yield "done", {"error": "The API key was rejected. Check ANTHROPIC_API_KEY in the app's secrets.", "cost": spent}
            return
        except anthropic.RateLimitError:
            yield "done", {"error": "Too many requests right now. Wait a minute and try again.", "cost": spent}
            return
        except anthropic.APIStatusError as e:
            yield "done", {"error": f"The Claude API returned an error ({e.status_code}). Try again shortly.", "cost": spent}
            return
        except anthropic.APIConnectionError:
            yield "done", {"error": "Could not reach the Claude API. Check the internet connection.", "cost": spent}
            return
        spent += cost(model, final.usage)

        if final.stop_reason == "refusal":
            yield "done", {"error": "The assistant declined to answer that. Try asking about the project's results.", "cost": spent, "drop_turn": True}
            return
        tool_uses = [b for b in final.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": final.content})
        if final.stop_reason == "max_tokens":
            yield "done", {"error": "The answer hit the length limit. Ask a narrower question.", "cost": spent, "truncated": bool(tool_uses)}
            return
        if final.stop_reason == "pause_turn":
            continue
        if not tool_uses:
            yield "done", {"cost": spent, "usage": final.usage}
            return
        results = []
        for b in tool_uses:
            yield "tool", f"{b.name}({', '.join(f'{k}={v}' for k, v in (b.input or {}).items())})"
            out, is_err = run_tool(b.name, b.input)
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": is_err})
        messages.append({"role": "user", "content": results})          # all results in one message
    yield "done", {"error": "The assistant needed too many lookups. Try a more specific question.", "cost": spent}


if __name__ == "__main__":       # quick offline check of every tool
    print(list_tables()[:600], "\n")
    print(read_table("staged_social_media.csv")[:400], "\n")
    print(search_questions("sleep"), "\n")
    print(question_breakdown("Q80")[:500], "\n")
    print(group_sad_rate("sex", "2025"), "\n")
    print(run_tool("group_sad_rate", {"group_by": "nope"}), run_tool("read_table", {}), sep="\n")
