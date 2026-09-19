import os
import re
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI

# ============================================================
# 1. APP CONFIG
# ============================================================

st.set_page_config(
    page_title="My AI agent",
    page_icon="🛒",
    layout="wide",
)

DB_PATH = Path("data/processed/ecommerce_clean.db").resolve()
CODEBOOK_PATH = Path("ecommerce_agent_codebook.md")

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

if not CODEBOOK_PATH.exists():
    st.error(f"Codebook not found: {CODEBOOK_PATH}")
    st.stop()

codebook_text = CODEBOOK_PATH.read_text(encoding="utf-8")


# ============================================================
# 2. SQL SAFETY
# ============================================================

FORBIDDEN_SQL = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "REPLACE",
    "ATTACH",
    "DETACH",
    "VACUUM",
    "REINDEX",
    "PRAGMA",
}

def validate_readonly_sql(sql: str):
    if not isinstance(sql, str) or not sql.strip():
        return False, "SQL query is empty."

    cleaned = sql.strip()
    first_word = cleaned.split()[0].upper()

    if first_word not in {"SELECT", "WITH"}:
        return False, "Only SELECT or WITH queries are allowed."

    for keyword in FORBIDDEN_SQL:
        if re.search(
            rf"\b{keyword}\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return False, f"Forbidden SQL keyword: {keyword}"

    without_last_semicolon = cleaned.rstrip(";")

    if ";" in without_last_semicolon:
        return False, "Multiple SQL statements are not allowed."

    return True, "OK"


def validate_sql_query(sql: str) -> dict:
    safe, reason = validate_readonly_sql(sql)

    if not safe:
        return {
            "status": "BLOCKED",
            "messages": [reason],
        }

    normalized = " ".join(sql.lower().split())

    try:
        db_uri = f"file:{DB_PATH}?mode=ro"

        with sqlite3.connect(db_uri, uri=True) as conn:
            conn.execute(
                "EXPLAIN QUERY PLAN " + sql
            ).fetchall()

    except Exception as e:
        return {
            "status": "BLOCKED",
            "messages": [
                f"SQL does not compile: {type(e).__name__}: {e}"
            ],
        }

    warnings = []

    if re.search(r"\bsum\s*\(\s*(?:\w+\.)?price\s*\)", normalized):
        warnings.append(
            "SUM(price) is the sum of price on retained order-item rows, "
            "not guaranteed complete original-order revenue."
        )

    if re.search(r"\b(sum|avg)\s*\(\s*(?:\w+\.)?payment_value\s*\)", normalized):
        warnings.append(
            "payment_value comes from one retained payment row per order; "
            "do not automatically interpret it as complete original-order revenue."
        )

    if ("product_category_name" in normalized and "payment_value" in normalized):
        warnings.append(
            "Category × payment_value is high-risk because category belongs "
            "to the retained product row while payment_value belongs to the "
            "retained payment row."
        )

    if (re.search(r"\bcount\s*\(", normalized) and "order_items" in normalized):
        warnings.append(
            "Counting order_items rows does not recover the original item count; "
            "this transformed table contains one retained row per order."
        )

    if (re.search(r"\bcount\s*\(", normalized) and "payments" in normalized):
        warnings.append(
            "Counting payments rows does not recover the original number of "
            "payment transactions; one retained payment row remains per order."
        )

    if ("order_delivered_timestamp" in normalized and "is not null" not in normalized):
        warnings.append(
            "Delivery analysis references order_delivered_timestamp without "
            "explicitly requiring a non-NULL actual-delivery timestamp."
        )

    if warnings:
        return {
            "status": "WARNING",
            "messages": warnings,
        }

    return {
        "status": "SAFE",
        "messages": ["SQL passed validation."],
    }


def execute_sql_dataframe(sql: str, max_rows: int = 200):
    validation = validate_sql_query(sql)

    if validation["status"] == "BLOCKED":
        raise ValueError(
            "SQL BLOCKED: "
            + " ".join(validation["messages"])
        )

    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(db_uri, uri=True) as conn:
        df = pd.read_sql_query(sql, conn)

    return df.head(max_rows), validation


# ============================================================
# 3. DATABASE CONTEXT
# ============================================================

@st.cache_data(show_spinner=False)
def build_schema_context() -> str:
    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(db_uri, uri=True) as conn:
        rows = conn.execute(
            """
            SELECT name, type, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name;
            """
        ).fetchall()

    parts = []
    for name, kind, sql in rows:
        parts.append(f"{kind.upper()}: {name}\n{sql}")

    return "\n\n".join(parts)


SCHEMA_CONTEXT = build_schema_context()


# ============================================================
# 4. LLM (GEMINI 1.5 FLASH)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

def llm_text(prompt: str) -> str:
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        raise ValueError("Vui lòng nhập Gemini API Key ở mục ⚙️ Cấu hình bên trái trước khi chat!")
        
    llm = get_llm(api_key)
    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts).strip()

    return str(content).strip()


def extract_json_payload(text: str):
    if not text:
        raise ValueError("Empty LLM response.")

    cleaned = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except Exception:
            pass

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start != -1 and object_end > object_start:
        try:
            return json.loads(cleaned[object_start:object_end + 1])
        except Exception:
            pass

    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    if array_start != -1 and array_end > array_start:
        try:
            return json.loads(cleaned[array_start:array_end + 1])
        except Exception:
            pass

    raise ValueError("Could not parse valid JSON from LLM response.")


def llm_json(prompt: str, required_key: Optional[str] = None, retries: int = 2):
    last_error = None
    raw = ""

    for attempt in range(retries + 1):
        try:
            if attempt == 0:
                raw = llm_text(prompt)
            else:
                repair_prompt = f"""
Repair the following malformed model output into VALID JSON only.

MALFORMED OUTPUT:
{raw}

Rules:
- Preserve the intended information.
- Return JSON only.
- No markdown fences.
- No commentary before or after JSON.
"""
                raw = llm_text(repair_prompt)

            data = extract_json_payload(raw)

            if required_key is not None and not (isinstance(data, dict) and required_key in data):
                raise ValueError(f"Missing required JSON key: {required_key}")

            return data

        except Exception as e:
            last_error = e

    raise RuntimeError(f"LLM JSON generation failed after repair attempts: {last_error}")


# ============================================================
# 5. COMMON HELPERS
# ============================================================

def format_history_for_prompt(history, max_messages=8):
    if not history:
        return "(no prior conversation)"
    lines = []
    for msg in history[-max_messages:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def normalize_sql_plan_item(item: Dict[str, Any], index: int):
    normalized = {
        "id": str(item.get("id", f"A{index}")),
        "title": str(item.get("title", f"Analysis {index}")),
        "reason": str(item.get("reason", "")),
        "sql": str(item.get("sql", "")).strip(),
    }

    def clean_label(text: str, metric_phrase: str):
        text = re.sub(r"\brevenue\b", metric_phrase, text, flags=re.IGNORECASE)
        text = re.sub(r"\bsales\b", metric_phrase, text, flags=re.IGNORECASE)
        duplicate = f"{metric_phrase} {metric_phrase}"
        while duplicate.lower() in text.lower():
            text = re.sub(re.escape(duplicate), metric_phrase, text, flags=re.IGNORECASE)
        text = re.sub(r"retained item-price retained item-price total", "retained item-price total", text, flags=re.IGNORECASE)
        text = re.sub(r"retained payment-value retained payment-value total", "retained payment-value total", text, flags=re.IGNORECASE)
        return text

    if re.search(r"\bsum\s*\(\s*(?:\w+\.)?price\s*\)", normalized["sql"], flags=re.IGNORECASE):
        normalized["title"] = clean_label(normalized["title"], "retained item-price total")
        normalized["reason"] = clean_label(normalized["reason"], "retained item-price total")

    if re.search(r"\bsum\s*\(\s*(?:\w+\.)?payment_value\s*\)", normalized["sql"], flags=re.IGNORECASE):
        normalized["title"] = clean_label(normalized["title"], "retained payment-value total")
        normalized["reason"] = clean_label(normalized["reason"], "retained payment-value total")

    return normalized


def run_sql_plan(plan_items: List[Dict[str, Any]], max_queries: int = 6):
    results = []
    for index, item in enumerate(plan_items[:max_queries], start=1):
        normalized = normalize_sql_plan_item(item, index)
        sql = normalized["sql"]
        validation = validate_sql_query(sql)

        result = {
            **normalized,
            "status": validation["status"],
            "validation_messages": validation["messages"],
            "records": [],
            "columns": [],
            "row_count_returned": 0,
        }

        if validation["status"] != "BLOCKED":
            try:
                df, _ = execute_sql_dataframe(sql, max_rows=200)
                result["records"] = df.to_dict(orient="records")
                result["columns"] = df.columns.tolist()
                result["row_count_returned"] = len(df)
            except Exception as e:
                result["status"] = "ERROR"
                result["validation_messages"].append(f"{type(e).__name__}: {e}")

        results.append(result)

    return results


def compact_evidence(analyses, max_rows_per_query=20):
    compact = []
    for item in analyses:
        compact.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "reason": item.get("reason"),
            "status": item.get("status"),
            "validation_messages": item.get("validation_messages", []),
            "sql": item.get("sql"),
            "rows": item.get("records", [])[:max_rows_per_query],
        })
    return compact


def clean_string_list(value):
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if item is not None and str(item).strip()]


def empty_strategy():
    return {"short_term": [], "medium_term": [], "long_term": []}


def normalize_strategy(value):
    if not isinstance(value, dict):
        return empty_strategy()
    return {
        "short_term": clean_string_list(value.get("short_term", [])),
        "medium_term": clean_string_list(value.get("medium_term", [])),
        "long_term": clean_string_list(value.get("long_term", [])),
    }


def no_chart():
    return {
        "type": "none",
        "x": None,
        "y": None,
        "title": None,
        "source_analysis_id": None,
    }


# ============================================================
# 6. STAGE 1 — ANALYST / PRIMARY SQL DISCOVERY
# ============================================================

def parse_tagged_analyses(text: str):
    analyses = []
    blocks = re.findall(r"<ANALYSIS>(.*?)</ANALYSIS>", text, flags=re.DOTALL | re.IGNORECASE)

    for index, block in enumerate(blocks[:4], start=1):
        def tag(name):
            match = re.search(rf"<{name}>(.*?)</{name}>", block, flags=re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        sql = tag("SQL")
        if not sql:
            continue

        # Cạo sạch markdown nếu Gemini bọc thêm
        sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s*```$", "", sql)

        analyses.append({
            "id": tag("ID") or f"A{index}",
            "title": tag("TITLE") or f"Analysis {index}",
            "reason": tag("REASON"),
            "sql": sql.strip(),
        })

    return analyses


def plan_primary_analyses(question: str, history):
    history_text = format_history_for_prompt(history)

    prompt = f"""
You are Stage 1: ANALYST in an AI Data Analyst system.

Your task is NOT to answer the user yet.
Your task is to decide what SQL evidence is needed.

USER QUESTION:
{question}

RECENT CONVERSATION:
{history_text}

DATABASE SCHEMA:
{SCHEMA_CONTEXT}

DATABASE CODEBOOK / SEMANTIC RULES:
{codebook_text}

RULES:
- Use only the actual schema above.
- Generate only read-only SELECT or WITH SQL.
- Never use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA.
- Every numerical claim later must come from SQL evidence.
- Respect transformed-data limitations in the codebook.
- Do not use unsupported basket-size, repeat-customer, original item-count,
  or original payment-transaction-count metrics.
- Never assume a currency.
- If you use SUM(price), call it "retained item-price total".
  NEVER call it revenue or sales.
- If you use SUM(payment_value), call it "retained payment-value total".
  NEVER call it guaranteed complete revenue.
- Do not describe payment_type counts as customer preference.
- For a narrow factual question, create 1-3 focused analyses.
- For an open-ended business/strategy/insight question, create EXACTLY 4
  complementary analyses chosen by YOU from the available schema.
- Choose analyses because they help answer THIS user's question.
- Do not pre-program a business story or paradox.
- Keep each SQL query reasonably compact.

Return ONLY this tagged format. Do not use markdown fences. Do not write anything before or after the tags.

<ANALYSIS>
<ID>A1</ID>
<TITLE>short descriptive title</TITLE>
<REASON>why this evidence is useful</REASON>
<SQL>SELECT ...</SQL>
</ANALYSIS>

Continue until the required number of analyses is provided.
"""

    raw = llm_text(prompt)
    analyses = parse_tagged_analyses(raw)

    if not analyses:
        repair_prompt = f"""
Reformat the following Analyst response into <ANALYSIS> blocks.
Preserve the intended analyses and SQL. Do not use markdown. Return only tagged blocks.

ORIGINAL RESPONSE:
{raw}

Required block format:
<ANALYSIS>
<ID>A1</ID>
<TITLE>...</TITLE>
<REASON>...</REASON>
<SQL>SELECT ...</SQL>
</ANALYSIS>
"""
        repaired = llm_text(repair_prompt)
        analyses = parse_tagged_analyses(repaired)

    if not analyses:
        raise RuntimeError("Analyst response could not be parsed into tagged SQL analyses.")

    return analyses


# ============================================================
# 7. STAGE 2 — PARADOX HUNTER
# ============================================================

def parse_tagged_paradox_tests(text: str):
    tests = []
    blocks = re.findall(r"<TEST>(.*?)</TEST>", text, flags=re.DOTALL | re.IGNORECASE)

    for index, block in enumerate(blocks[:3], start=1):
        def tag(name):
            match = re.search(rf"<{name}>(.*?)</{name}>", block, flags=re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        sql = tag("SQL")
        if not sql:
            continue

        # Cạo sạch markdown nếu Gemini bọc thêm
        sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s*```$", "", sql)

        tests.append({
            "id": tag("ID") or f"T{index}",
            "hypothesis": tag("QUESTION"),
            "why_surprising": tag("WHY"),
            "verification_logic": (
                "Judge the returned comparison table and report only "
                "a contrast/reversal/tension directly supported by its rows."
            ),
            "verification_sql": sql.strip(),
            "round_number": 1,
        })

    return tests


def plan_fast_paradox_tests(question: str, primary_results):
    evidence = compact_evidence(primary_results, max_rows_per_query=10)

    prompt = f"""
You are the PARADOX HUNTER in an AI Data Analyst system.

USER QUESTION:
{question}

PRIMARY SQL EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, default=str)}

DATABASE SCHEMA:
{SCHEMA_CONTEXT}

DATA LIMITATIONS:
- orders: one row per order.
- order_items: one retained row per order.
- payments: one retained payment row per order.
- products: one metadata row per product_id.
- customers: transformed unique customer rows.
- SUM(price) = retained item-price total only.
- SUM(payment_value) = retained payment-value total only.

TASK:
Create EXACTLY 3 exploratory, falsifiable paradox tests unless the schema
literally makes three distinct comparisons impossible.

Return EXACTLY this tagged format, with no markdown fences and no text outside it:

<TEST>
<ID>T1</ID>
<QUESTION>broad testable business question</QUESTION>
<WHY>why a surprising result would matter</WHY>
<SQL>WITH ... SELECT ...</SQL>
</TEST>
"""
    raw = llm_text(prompt)
    tests = parse_tagged_paradox_tests(raw)

    if not tests:
        repair_prompt = f"""
Reformat the following response into EXACTLY 3 <TEST> blocks.
Preserve its intended hypotheses and SQL where possible. Do not add markdown or commentary.

ORIGINAL RESPONSE:
{raw}

Required format:
<TEST>
<ID>T1</ID>
<QUESTION>...</QUESTION>
<WHY>...</WHY>
<SQL>...</SQL>
</TEST>
"""
        repaired = llm_text(repair_prompt)
        tests = parse_tagged_paradox_tests(repaired)

    return tests[:3]


def execute_fast_paradox_tests(tests):
    results = []
    for item in tests[:3]:
        sql = item.get("verification_sql", "")
        validation = validate_sql_query(sql)

        result = {
            **item,
            "status": validation["status"],
            "validation_messages": validation["messages"],
            "records": [],
            "columns": [],
        }

        if validation["status"] != "BLOCKED":
            try:
                df, _ = execute_sql_dataframe(sql, max_rows=120)
                result["records"] = df.to_dict(orient="records")
                result["columns"] = df.columns.tolist()
            except Exception as e:
                result["status"] = "ERROR"
                result["validation_messages"].append(f"{type(e).__name__}: {e}")

        results.append(result)

    return results


# ============================================================
# 7D. FAST FINAL JUDGE + STRATEGIST (v7)
# ============================================================

def parse_items_from_block(text: str, block_name: str):
    block_match = re.search(rf"<{block_name}>(.*?)</{block_name}>", text, flags=re.DOTALL | re.IGNORECASE)
    if not block_match:
        return []
    return [
        item.strip() for item in re.findall(r"<ITEM>(.*?)</ITEM>", block_match.group(1), flags=re.DOTALL | re.IGNORECASE) if item.strip()
    ]


def parse_final_tagged_report(text: str):
    def single_tag(name):
        match = re.search(rf"<{name}>(.*?)</{name}>", text, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    answer = single_tag("ANSWER")
    basic_insights = parse_items_from_block(text, "BASIC_INSIGHTS")
    paradoxical_insights = parse_items_from_block(text, "PARADOXICAL_INSIGHTS")
    short_term = parse_items_from_block(text, "SHORT_TERM")
    medium_term = parse_items_from_block(text, "MEDIUM_TERM")
    long_term = parse_items_from_block(text, "LONG_TERM")
    limitations = parse_items_from_block(text, "LIMITATIONS")

    judgments = []
    judgments_block = re.search(r"<JUDGMENTS>(.*?)</JUDGMENTS>", text, flags=re.DOTALL | re.IGNORECASE)
    if judgments_block:
        judgment_blocks = re.findall(r"<JUDGMENT>(.*?)</JUDGMENT>", judgments_block.group(1), flags=re.DOTALL | re.IGNORECASE)
        for block in judgment_blocks:
            def local_tag(name):
                match = re.search(rf"<{name}>(.*?)</{name}>", block, flags=re.DOTALL | re.IGNORECASE)
                return match.group(1).strip() if match else ""

            supported_text = local_tag("SUPPORTED").strip().lower()
            judgments.append({
                "id": local_tag("ID"),
                "supported": (supported_text in {"true", "yes", "1"}),
                "insight": local_tag("INSIGHT"),
                "reason": local_tag("REASON"),
            })

    if not answer:
        raise ValueError("Missing <ANSWER> in final report.")

    return {
        "answer": answer,
        "basic_insights": basic_insights,
        "paradoxical_insights": paradoxical_insights,
        "strategy": {
            "short_term": short_term,
            "medium_term": medium_term,
            "long_term": long_term,
        },
        "limitations": limitations,
        "judgments": judgments,
        "chart": no_chart(),
    }


def fast_final_report(question: str, primary_results, paradox_results, chart_requested: bool):
    primary_evidence = compact_evidence(primary_results, max_rows_per_query=8)
    paradox_evidence = []
    for item in paradox_results:
        paradox_evidence.append({
            "id": item.get("id"),
            "test_question": item.get("hypothesis"),
            "why_interesting": item.get("why_surprising"),
            "status": item.get("status"),
            "validation_messages": item.get("validation_messages", []),
            "sql": item.get("verification_sql"),
            "rows": item.get("records", [])[:25],
        })

    prompt = f"""
You are the FINAL DATA JUDGE AND BUSINESS STRATEGIST.

USER QUESTION:
{question}

PRIMARY EXECUTED SQL EVIDENCE:
{json.dumps(primary_evidence, ensure_ascii=False, default=str)}

PARADOX TESTS + EXECUTED RESULTS:
{json.dumps(paradox_evidence, ensure_ascii=False, default=str)}

Return ONLY the following tagged format. Do not use markdown fences. Do not write anything outside these tags.

<ANSWER>
...
</ANSWER>

<BASIC_INSIGHTS>
<ITEM>...</ITEM>
</BASIC_INSIGHTS>

<JUDGMENTS>
<JUDGMENT>
<ID>T1</ID>
<SUPPORTED>true</SUPPORTED>
<INSIGHT>...</INSIGHT>
<REASON>...</REASON>
</JUDGMENT>
</JUDGMENTS>

<PARADOXICAL_INSIGHTS>
<ITEM>...</ITEM>
</PARADOXICAL_INSIGHTS>

<SHORT_TERM>
<ITEM>...</ITEM>
</SHORT_TERM>

<MEDIUM_TERM>
<ITEM>...</ITEM>
</MEDIUM_TERM>

<LONG_TERM>
<ITEM>...</ITEM>
</LONG_TERM>

<LIMITATIONS>
<ITEM>...</ITEM>
</LIMITATIONS>
"""
    raw = llm_text(prompt)

    try:
        report = parse_final_tagged_report(raw)
    except Exception:
        repair_prompt = f"""
Reformat the following analyst response into the exact tagged report format.

ORIGINAL:
{raw}

Required tags:
<ANSWER>...</ANSWER>
<BASIC_INSIGHTS><ITEM>...</ITEM></BASIC_INSIGHTS>
<JUDGMENTS><JUDGMENT><ID>T1</ID><SUPPORTED>true|false</SUPPORTED><INSIGHT>...</INSIGHT><REASON>...</REASON></JUDGMENT></JUDGMENTS>
<PARADOXICAL_INSIGHTS><ITEM>...</ITEM></PARADOXICAL_INSIGHTS>
<SHORT_TERM><ITEM>...</ITEM></SHORT_TERM>
<MEDIUM_TERM><ITEM>...</ITEM></MEDIUM_TERM>
<LONG_TERM><ITEM>...</ITEM></LONG_TERM>
<LIMITATIONS><ITEM>...</ITEM></LIMITATIONS>
"""
        repaired = llm_text(repair_prompt)
        report = parse_final_tagged_report(repaired)

    report["chart"] = no_chart()
    return report


# ============================================================
# 10. CHART HELPERS
# ============================================================

def detect_chart_requested(question: str):
    q = question.lower()
    keywords = ["chart", "graph", "plot", "visual", "visualization", "biểu đồ", "vẽ"]
    return any(word in q for word in keywords)

def deterministic_chart_fallback(chart, primary_results, chart_requested):
    if not chart_requested:
        return no_chart()

    for item in primary_results:
        records = item.get("records", [])
        if not records:
            continue
        df = pd.DataFrame(records)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = [col for col in df.columns if col not in numeric_cols]

        if categorical_cols and numeric_cols:
            return {
                "type": "bar",
                "x": categorical_cols[0],
                "y": numeric_cols[0],
                "title": item.get("title"),
                "source_analysis_id": item.get("id"),
            }
        if len(numeric_cols) >= 2:
            return {
                "type": "scatter",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
                "title": item.get("title"),
                "source_analysis_id": item.get("id"),
            }
    return no_chart()


# ============================================================
# 11. ERROR RESULTS
# ============================================================

def error_result(message: str, stage_trace=None):
    return {
        "answer": message,
        "primary_analyses": [],
        "paradox_candidates": [],
        "paradox_judgments": [],
        "basic_insights": [],
        "paradoxical_insights": [],
        "strategy": empty_strategy(),
        "limitations": ["The agent workflow did not complete."],
        "chart": no_chart(),
        "stage_trace": stage_trace or {},
        "error": True,
    }


# ============================================================
# 12. FULL AGENT WORKFLOW
# ============================================================

def ask_data_agent(question: str, history):
    # CHẶN LỖI: Yêu cầu API Key ngay từ vòng gửi xe
    if not st.session_state.get("api_key"):
        return error_result("⚠️ Cảnh báo: Bạn chưa nhập Gemini API Key! Vui lòng mở mục '⚙️ Cấu hình' ở thanh Sidebar bên trái để nhập chìa khóa trước khi chat.")

    stage_trace = {
        "analyst": "RUNNING",
        "paradox_hunter": "PENDING",
        "paradox_verification": "PENDING",
        "strategist": "PENDING",
    }

    try:
        plan = plan_primary_analyses(question, history)
    except Exception as e:
        stage_trace["analyst"] = f"FAILED — {type(e).__name__}: {e}"
        return error_result("Analyst failed while planning SQL. The model response could not be converted into executable analysis blocks.", stage_trace)

    primary_results = run_sql_plan(plan, max_queries=4)
    usable_primary = [i for i in primary_results if i.get("status") in {"SAFE", "WARNING"} and i.get("records")]

    if not usable_primary:
        stage_trace["analyst"] = "FAILED"
        return error_result("Analyst could not obtain usable SQL evidence. Inspect the SQL tab for generated queries.", stage_trace)

    stage_trace["analyst"] = f"COMPLETED — {len(usable_primary)} usable analysis table(s)"
    stage_trace["paradox_hunter"] = "RUNNING"

    try:
        paradox_tests = plan_fast_paradox_tests(question, usable_primary)
        stage_trace["paradox_hunter"] = f"COMPLETED — {len(paradox_tests)} broad paradox test(s)"
    except Exception as e:
        paradox_tests = []
        stage_trace["paradox_hunter"] = f"FAILED — {type(e).__name__}: {e}"

    stage_trace["paradox_verification"] = "RUNNING"
    paradox_results = execute_fast_paradox_tests(paradox_tests)
    usable_tests = [i for i in paradox_results if i.get("status") in {"SAFE", "WARNING"} and i.get("records")]
    stage_trace["paradox_verification"] = f"COMPLETED — {len(usable_tests)} test table(s) returned evidence"

    stage_trace["strategist"] = "RUNNING"
    chart_requested = detect_chart_requested(question)

    try:
        report = fast_final_report(question, usable_primary, usable_tests, chart_requested)
        judgments = report.get("judgments", [])
        supported_count = sum(1 for item in judgments if item.get("supported") is True)
        stage_trace["paradox_verification"] += f" | {supported_count} paradox(es) verified"
        stage_trace["strategist"] = "COMPLETED"
    except Exception as e:
        judgments = []
        stage_trace["strategist"] = f"FAILED — {type(e).__name__}: {e}"
        report = {
            "answer": "SQL evidence đã được thu thập thành công, nhưng bước tổng hợp ngôn ngữ cuối cùng gặp lỗi. Evidence vẫn có thể kiểm tra ở các tab bên dưới.",
            "basic_insights": [],
            "paradoxical_insights": ["Chưa thể kết luận insight nghịch lý vì Final Judge không hoàn tất."],
            "strategy": empty_strategy(),
            "limitations": [],
            "chart": no_chart(),
        }

    chart = deterministic_chart_fallback(report.get("chart", no_chart()), usable_primary, chart_requested)

    return {
        "answer": report.get("answer", ""),
        "primary_analyses": primary_results,
        "paradox_candidates": paradox_results,
        "paradox_judgments": judgments,
        "basic_insights": report.get("basic_insights", []),
        "paradoxical_insights": report.get("paradoxical_insights", []),
        "strategy": report.get("strategy", empty_strategy()),
        "limitations": report.get("limitations", []),
        "chart": chart,
        "stage_trace": stage_trace,
        "error": False,
    }


# ============================================================
# 13. MULTI-CHAT SESSION STATE (LƯU TRỮ VĨNH VIỄN)
# ============================================================

HISTORY_FILE = "chat_history_v7.json"

def load_chats():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_chats():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chats, f, ensure_ascii=False, indent=4)

def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [],
    }
    st.session_state.current_chat_id = chat_id
    save_chats()

def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
    
    if not st.session_state.chats:
        create_new_chat()
    elif st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = next(reversed(st.session_state.chats))
    
    save_chats()

def clear_all_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.session_state.chats = {}
    create_new_chat()

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if not st.session_state.chats:
    create_new_chat()
elif "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[-1]


# ============================================================
# 14. SIDEBAR — GIAO DIỆN CHUẨN Y HỆT ẢNH
# ============================================================

with st.sidebar:
    st.markdown("### 🔥 Group 3 - TINE313")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Chat Mới", type="primary", use_container_width=True):
            create_new_chat()
            st.rerun()
    with col2:
        with st.popover("⚙️ Cấu hình", use_container_width=True):
            st.text_input("Gemini API Key:", type="password", key="api_key")
            st.markdown("[👉 Lấy API Key tại đây](https://aistudio.google.com/app/apikey)")
            st.markdown("---")
            st.markdown("**Kết nối MySQL Workbench**")
            st.caption("Để trống nếu dùng CSDL SQLite mặc định.")
            st.text_input("Host (VD: localhost):", key="db_host")
            st.text_input("Username (VD: root):", key="db_user")
            st.text_input("Password:", type="password", key="db_pass")
            st.text_input("Database Name:", key="db_name")

    st.markdown("---")
    st.markdown("📂 **Danh mục Bảng Dữ liệu**")
    with st.expander("Hiển thị chi tiết bảng"):
        st.markdown("""
        - **df_customers** (Khách hàng)
        - **df_orders** (Đơn hàng trung tâm)
        - **df_orderitems** (Chi tiết giao hàng)
        - **df_products** (Sản phẩm)
        - **df_payments** (Thanh toán)
        """)

    st.markdown("---")
    st.markdown("🕒 **Lịch sử Hội thoại**")
    
    has_history = False
    for chat_id, chat in reversed(list(st.session_state.chats.items())):
        if len(chat.get("messages", [])) > 0:
            has_history = True
            cols = st.columns([0.82, 0.18])
            
            is_current = (chat_id == st.session_state.current_chat_id)
            title = chat.get("title", "New chat")
            label = f"👉 {title}" if is_current else f"💬 {title}"
            
            with cols[0]:
                if st.button(label, key=f"open_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            with cols[1]:
                if st.button("🗑️", key=f"delete_{chat_id}", help="Xóa chat này", use_container_width=True):
                    delete_chat(chat_id)
                    st.rerun()
                    
    if not has_history:
        st.info("Chưa có lịch sử trò chuyện.")
        
    if st.button("🗑️ Dọn dẹp TOÀN BỘ lịch sử", use_container_width=True):
        clear_all_history()
        st.rerun()

current_chat = st.session_state.chats[st.session_state.current_chat_id]
history = current_chat["messages"]


# ============================================================
# 15. UI HELPERS
# ============================================================

def render_bullets(items, empty_text="No evidence-based item available."):
    if not items:
        st.info(empty_text)
        return
    for item in items:
        if item:
            st.markdown(f"- {item}")


def analysis_to_df(item):
    return pd.DataFrame(item.get("records", []))


def get_chart_df(result):
    chart = result.get("chart", {})
    source_id = chart.get("source_analysis_id")
    if not source_id: return None
    for item in result.get("primary_analyses", []):
        if item.get("id") == source_id:
            return analysis_to_df(item)
    return None


def render_chart(result):
    chart = result.get("chart", no_chart())
    chart_type = chart.get("type", "none")
    if chart_type == "none": return
    
    df = get_chart_df(result)
    if df is None or df.empty: return

    x = chart.get("x")
    y = chart.get("y")
    title = chart.get("title")

    if x not in df.columns or y not in df.columns: return

    fig = None
    if chart_type == "bar":
        fig = px.bar(df, x=x, y=y, title=title)
    elif chart_type == "line":
        fig = px.line(df, x=x, y=y, title=title, markers=True)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, title=title)
    elif chart_type == "pie":
        fig = px.pie(df, names=x, values=y, title=title)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


def render_stage_audit(result):
    trace = result.get("stage_trace", {})
    with st.expander("🔍 Auto-Audit & Agent Workflow", expanded=False):
        st.markdown("#### Agent stages")
        st.markdown(f"- **1. Analyst:** {trace.get('analyst', 'UNKNOWN')}")
        st.markdown(f"- **2. Paradox Hunter:** {trace.get('paradox_hunter', 'UNKNOWN')}")
        st.markdown(f"- **3. Paradox Verification/Judge:** {trace.get('paradox_verification', 'UNKNOWN')}")
        st.markdown(f"- **4. Business Strategist:** {trace.get('strategist', 'UNKNOWN')}")
        
        st.markdown("---")
        st.markdown("#### SQL validation summary")
        
        all_items = result.get("primary_analyses", []) + result.get("paradox_candidates", [])
        if not all_items:
            st.info("No SQL was executed.")
            return

        safe = sum(1 for item in all_items if item.get("status") == "SAFE")
        warning = sum(1 for item in all_items if item.get("status") == "WARNING")
        blocked = sum(1 for item in all_items if item.get("status") == "BLOCKED")

        st.write(f"SAFE: {safe} | WARNING: {warning} | BLOCKED: {blocked}")


def render_evidence_tab(result):
    st.markdown("### Primary evidence")
    primary = result.get("primary_analyses", [])
    if not primary: st.info("No primary SQL evidence.")
    
    for item in primary:
        st.markdown(f"#### {item.get('id')} — {item.get('title')}")
        if item.get("reason"): st.caption(item["reason"])
        st.caption(f"SQL validation: {item.get('status')}")
        df = analysis_to_df(item)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Paradox discovery & verification")
    candidates = result.get("paradox_candidates", [])
    judgments = {str(item.get("id")): item for item in result.get("paradox_judgments", [])}

    if not candidates:
        st.info("Paradox Hunter did not produce a testable candidate in this run.")

    for item in candidates:
        candidate_id = str(item.get("id"))
        judgment = judgments.get(candidate_id, {})

        st.markdown(f"#### {candidate_id}")
        st.write(item.get("hypothesis", ""))
        if item.get("why_surprising"):
            st.caption("Why it may be surprising: " + item["why_surprising"])

        df = pd.DataFrame(item.get("records", []))
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)

        supported = judgment.get("supported")
        if supported is True:
            st.success("VERIFIED — " + str(judgment.get("reason", "")))
        elif supported is False:
            st.warning("NOT VERIFIED — " + str(judgment.get("reason", "")))
        else:
            st.info("No judge decision available.")


def render_sql_tab(result):
    st.markdown("### Analyst SQL")
    for item in result.get("primary_analyses", []):
        st.markdown(f"#### {item.get('id')} — {item.get('status')}")
        st.code(item.get("sql", ""), language="sql")
        render_bullets(item.get("validation_messages", []), "")

    st.markdown("---")
    st.markdown("### Paradox verification SQL")
    for item in result.get("paradox_candidates", []):
        st.markdown(f"#### {item.get('id')} — {item.get('status')}")
        st.code(item.get("verification_sql", ""), language="sql")
        render_bullets(item.get("validation_messages", []), "")


def render_report(result):
    if result.get("error"):
        st.error(result.get("answer", "Agent workflow failed."))
        render_stage_audit(result)
        return

    st.success("💡 AI Agent đã hoàn tất: phân tích → tìm nghịch lý → kiểm chứng bằng SQL → đề xuất chiến lược.")
    render_stage_audit(result)

    insight_tab, strategy_tab, evidence_tab, sql_tab = st.tabs([
        "📊 Báo cáo Insight", "💡 Chiến lược", "🧪 Evidence & Paradox Test", "⚙️ SQL",
    ])

    with insight_tab:
        st.markdown("### Kết luận")
        st.markdown(result.get("answer", ""))
        render_chart(result)
        
        st.markdown("### 1. Insight cơ bản")
        render_bullets(result.get("basic_insights", []), "No basic insight was produced.")
        
        st.markdown("### 2. Insight nghịch lý")
        render_bullets(result.get("paradoxical_insights", []), "No verified paradox was produced.")

        limitations = result.get("limitations", [])
        if limitations:
            with st.expander("⚠️ Giới hạn diễn giải"):
                render_bullets(limitations)

    with strategy_tab:
        strategy = result.get("strategy", empty_strategy())
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### ⚡ Ngắn hạn")
            render_bullets(strategy.get("short_term", []), "No short-term recommendation.")
        with col2:
            st.markdown("### 🧭 Trung hạn")
            render_bullets(strategy.get("medium_term", []), "No medium-term recommendation.")
        with col3:
            st.markdown("### 🏗️ Dài hạn")
            render_bullets(strategy.get("long_term", []), "No long-term recommendation.")

    with evidence_tab:
        render_evidence_tab(result)

    with sql_tab:
        render_sql_tab(result)


def render_message(message):
    role = message.get("role", "assistant")
    with st.chat_message(role):
        if role == "user":
            st.markdown(message.get("content", ""))
            return

        result = message.get("result")
        if result:
            render_report(result)
        else:
            st.markdown(message.get("content", ""))


def make_history_for_agent(messages):
    converted = []
    for message in messages[-10:]:
        converted.append({
            "role": message.get("role", "assistant"),
            "content": message.get("content", ""),
        })
    return converted


# ============================================================
# 16. MAIN PAGE
# ============================================================

st.title("🛒 My AI agent")
st.markdown("Trợ lý AI phân tích dữ liệu, săn Insight & Hoạch định Chiến lược\n\n🔥 **Agent phát triển bởi: Group 3 - TINE313** 🔥")

for message in history:
    render_message(message)

question = st.chat_input("VD: Cho tôi insights về địa lý...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    prior_history = make_history_for_agent(history)

    with st.spinner("Agent đang phân tích và kiểm chứng dữ liệu..."):
        result = ask_data_agent(
            question=question,
            history=prior_history,
        )

    history.append({
        "role": "user",
        "content": question,
    })

    history.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "result": result,
    })

    if current_chat.get("title") == "New chat":
        clean_title = " ".join(question.strip().split())
        if len(clean_title) > 34:
            clean_title = clean_title[:34] + "..."
        current_chat["title"] = clean_title

    # LƯU CHAT TRƯỚC KHI RERUN
    save_chats()
    st.rerun()