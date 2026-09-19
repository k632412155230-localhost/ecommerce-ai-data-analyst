import re
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from langchain_groq import ChatGroq


# ============================================================
# 1. APP CONFIG
# ============================================================

st.set_page_config(
    page_title="E-commerce AI Data Analyst",
    page_icon="📊",
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
    """
    Deterministic validator.
    It does NOT decide what the insight/paradox should be.
    """
    safe, reason = validate_readonly_sql(sql)

    if not safe:
        return {
            "status": "BLOCKED",
            "messages": [reason],
        }

    normalized = " ".join(sql.lower().split())

    # Compile against the actual SQLite database without executing.
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

    if re.search(
        r"\bsum\s*\(\s*(?:\w+\.)?price\s*\)",
        normalized,
    ):
        warnings.append(
            "SUM(price) is the sum of price on retained order-item rows, "
            "not guaranteed complete original-order revenue."
        )

    if re.search(
        r"\b(sum|avg)\s*\(\s*(?:\w+\.)?payment_value\s*\)",
        normalized,
    ):
        warnings.append(
            "payment_value comes from one retained payment row per order; "
            "do not automatically interpret it as complete original-order revenue."
        )

    if (
        "product_category_name" in normalized
        and "payment_value" in normalized
    ):
        warnings.append(
            "Category × payment_value is high-risk because category belongs "
            "to the retained product row while payment_value belongs to the "
            "retained payment row."
        )

    if (
        re.search(r"\bcount\s*\(", normalized)
        and "order_items" in normalized
    ):
        warnings.append(
            "Counting order_items rows does not recover the original item count; "
            "this transformed table contains one retained row per order."
        )

    if (
        re.search(r"\bcount\s*\(", normalized)
        and "payments" in normalized
    ):
        warnings.append(
            "Counting payments rows does not recover the original number of "
            "payment transactions; one retained payment row remains per order."
        )

    if (
        "order_delivered_timestamp" in normalized
        and "is not null" not in normalized
    ):
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


def execute_sql_dataframe(
    sql: str,
    max_rows: int = 200,
):
    validation = validate_sql_query(sql)

    if validation["status"] == "BLOCKED":
        raise ValueError(
            "SQL BLOCKED: "
            + " ".join(validation["messages"])
        )

    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(
        db_uri,
        uri=True,
    ) as conn:
        df = pd.read_sql_query(
            sql,
            conn,
        )

    return df.head(max_rows), validation


# ============================================================
# 3. DATABASE CONTEXT
# ============================================================

@st.cache_data(show_spinner=False)
def build_schema_context() -> str:
    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(
        db_uri,
        uri=True,
    ) as conn:
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
        parts.append(
            f"{kind.upper()}: {name}\n{sql}"
        )

    return "\n\n".join(parts)


SCHEMA_CONTEXT = build_schema_context()


# ============================================================
# 4. LLM
# ============================================================

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0,
    )


def llm_text(prompt: str) -> str:
    """
    Plain model call with NO tool binding.
    This avoids malformed tool-call JSON from the provider.
    """
    llm = get_llm()
    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, str):
        return content.strip()

    # Some providers may return structured content blocks.
    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(
                        str(item["text"])
                    )
            else:
                text_parts.append(str(item))

        return "\n".join(text_parts).strip()

    return str(content).strip()


def extract_json_payload(text: str):
    """
    Parse JSON robustly from:
    - raw JSON
    - ```json ... ```
    - prose followed by a JSON object/array
    """
    if not text:
        raise ValueError("Empty LLM response.")

    cleaned = text.strip()

    # Fenced JSON
    fenced = re.search(
        r"```(?:json)?\s*(.*?)