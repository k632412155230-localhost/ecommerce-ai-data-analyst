import json
import re
import sqlite3
import uuid
import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="My AI agent", page_icon="🛒", layout="wide")

DB_PATH = Path("data/processed/ecommerce_clean.db").resolve()
if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

# ---------- Compact schema ----------
@st.cache_data(show_spinner=False)
def compact_schema():
    uri = f"file:{DB_PATH}?mode=ro"
    out = []
    with sqlite3.connect(uri, uri=True) as conn:
        items = conn.execute("""
            SELECT name, type FROM sqlite_master
            WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
        """).fetchall()
        for name, kind in items:
            cols = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
            out.append(f"{kind.upper()} {name}: " + ", ".join(f"{c[1]} {c[2]}" for c in cols))
    return "\n".join(out)

SCHEMA = compact_schema()

SEMANTICS = """
DATA SEMANTICS
- orders: one row per order.
- order_items: one retained row per order in this transformed dataset.
- payments: one retained payment row per order.
- products: one metadata row per product_id.
- customers: one transformed unique customer row.
- Do not infer original basket size, original item count, original payment count, or repeat-customer behavior.
- SUM(price) = retained item-price total only; never call it complete revenue/sales.
- SUM(payment_value) = retained payment-value total only; never call it complete revenue.
- Payment-type frequency is not automatically customer preference.
- Never invent currency, profitability, demand, or causality.
"""

# ---------- SQL safety ----------
FORBIDDEN = {"INSERT","UPDATE","DELETE","DROP","ALTER","CREATE","REPLACE","ATTACH","DETACH","VACUUM","REINDEX","PRAGMA"}

def validate_sql(sql):
    if not isinstance(sql, str) or not sql.strip():
        return {"status":"BLOCKED","messages":["SQL is empty."]}
    s = sql.strip()
    if s.split()[0].upper() not in {"SELECT","WITH"}:
        return {"status":"BLOCKED","messages":["Only SELECT/WITH SQL is allowed."]}
    for kw in FORBIDDEN:
        if re.search(rf"\b{kw}\b", s, flags=re.I):
            return {"status":"BLOCKED","messages":[f"Forbidden SQL keyword: {kw}"]}
    if ";" in s.rstrip(";"):
        return {"status":"BLOCKED","messages":["Multiple SQL statements are not allowed."]}
    try:
        uri = f"file:{DB_PATH}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.execute("EXPLAIN QUERY PLAN " + s).fetchall()
    except Exception as e:
        return {"status":"BLOCKED","messages":[f"SQL compile error: {type(e).__name__}: {e}"]}

    low = " ".join(s.lower().split())
    warnings = []
    if re.search(r"\bsum\s*\(\s*(?:\w+\.)?price\s*\)", low):
        warnings.append("SUM(price) is retained item-price total, not guaranteed complete revenue.")
    if re.search(r"\bsum\s*\(\s*(?:\w+\.)?payment_value\s*\)", low):
        warnings.append("SUM(payment_value) is retained payment-value total, not guaranteed complete revenue.")
    if "order_delivered_timestamp" in low and "is not null" not in low:
        warnings.append("Delivery analysis should normally require order_delivered_timestamp IS NOT NULL.")

    if (
        "order_items" in low
        and (
            "avg_items_per_order" in low
            or "items_per_order" in low
            or "basket_size" in low
            or "total_items" in low
        )
    ):
        return {
            "status":"BLOCKED",
            "messages":[
                "Semantic block: transformed order_items has one retained row per order, "
                "so original item-count / basket-size / items-per-order metrics are unavailable."
            ],
        }

    return {"status":"WARNING" if warnings else "SAFE","messages":warnings or ["SQL passed validation."]}

def execute_sql(sql, max_rows=80):
    check = validate_sql(sql)
    if check["status"] == "BLOCKED":
        return pd.DataFrame(), check
    uri = f"file:{DB_PATH}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        df = pd.read_sql_query(sql, conn)
    return df.head(max_rows), check

# ---------- LLM (CHUẨN GEMINI 3.6 FLASH + API KEY) ----------
def llm():
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.error("⚠️ Vui lòng nhập Gemini API Key trong mục ⚙️ Cấu hình ở thanh bên trái!")
        st.stop()
        
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0,
        max_retries=1,
    )

def ask_llm(prompt):
    r = llm().invoke(prompt)
    c = r.content
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        return "\n".join(str(x.get("text","") if isinstance(x,dict) else x) for x in c).strip()
    return str(c).strip()

def is_rate_limit(e):
    x = str(e).lower()
    return "429" in x or "rate limit" in x or "rate_limit" in x

def friendly_error(e, stage):
    if is_rate_limit(e):
        m = re.search(r"try again in ([^.'}]+)", str(e), flags=re.I)
        wait = m.group(1).strip() if m else "a few minutes"
        return f"Groq rate limit reached during **{stage}**. Retry in about **{wait}**. No data was modified."
    return f"{stage} failed: {type(e).__name__}: {e}"

# ---------- Tolerant parsers ----------
def sql_blocks(text):
    tick3 = "`" * 3
    blocks = re.findall(rf"{tick3}sql\s*(.*?){tick3}", text, flags=re.S|re.I)
    if not blocks:
        blocks = re.findall(rf"{tick3}\s*((?:SELECT|WITH)\b.*?){tick3}", text, flags=re.S|re.I)
    return [x.strip() for x in blocks if x.strip()]

def parse_plan(text, prefix, limit):
    blocks = sql_blocks(text)
    heads = re.findall(rf"^##\s*({prefix}\d+)\s*\|\s*(.+?)\s*$", text, flags=re.M|re.I)
    out = []
    for i, sql in enumerate(blocks[:limit], 1):
        ident, title = heads[i-1] if i <= len(heads) else (f"{prefix}{i}", f"Analysis {i}")
        out.append({"id":ident.upper(),"title":title.strip(),"sql":sql})
    return out

HEADINGS = ["CONCLUSION","BASIC INSIGHTS","TEST JUDGMENTS","PARADOXICAL INSIGHTS","SHORT TERM","MEDIUM TERM","LONG TERM","LIMITATIONS"]

def get_section(text, heading):
    pat = rf"^##\s*{re.escape(heading)}\s*$\s*(.*?)(?=^##\s*(?:{'|'.join(map(re.escape,HEADINGS))})\s*$|\Z)"
    m = re.search(pat, text, flags=re.S|re.M|re.I)
    return m.group(1).strip() if m else ""

def bullets(text):
    return [x.strip() for x in re.findall(r"^\s*[-*]\s+(.+?)\s*$", text, flags=re.M) if x.strip()]

def parse_final(text):
    ans = get_section(text,"CONCLUSION") or text.strip()
    judgments = []
    for line in get_section(text,"TEST JUDGMENTS").splitlines():
        m = re.match(r"\s*[-*]?\s*\[(T\d+)\]\s*(SUPPORTED|NOT SUPPORTED)\s*[:\-]\s*(.*)", line, flags=re.I)
        if m:
            judgments.append({
                "id":m.group(1).upper(),
                "supported":m.group(2).upper()=="SUPPORTED",
                "reason":m.group(3).strip(),
            })
    return {
        "answer":ans,
        "basic_insights":bullets(get_section(text,"BASIC INSIGHTS")),
        "paradoxical_insights":bullets(get_section(text,"PARADOXICAL INSIGHTS")) or ["Không có insight nghịch lý đủ chắc được xác minh trong lượt này."],
        "strategy":{
            "short_term":bullets(get_section(text,"SHORT TERM")),
            "medium_term":bullets(get_section(text,"MEDIUM TERM")),
            "long_term":bullets(get_section(text,"LONG TERM")),
        },
        "limitations":bullets(get_section(text,"LIMITATIONS")),
        "judgments":judgments,
    }

# ---------- Stage 1 ----------
def analyst_plan(question, history_text):
    tick3 = "`" * 3
    prompt = f"""
You are the ANALYST stage.

USER QUESTION:
{question}

RECENT CONTEXT:
{history_text}

DATABASE:
{SCHEMA}

{SEMANTICS}

For an open-ended business insight/strategy request, create exactly 4 complementary SQL analyses.
For a narrow question, create 1-3.
Choose analyses yourself from the schema. Do not answer the user yet.

Rules:
- SELECT/WITH only.
- SQLITE syntax only.
- Use ONLY tables and columns that literally appear in DATABASE above.
- Before writing each query, verify every table name and column name against DATABASE.
- Do NOT invent generic columns such as category, segment, order_date, revenue, quantity.
- SQLite does NOT support DATE_TRUNC; use strftime() for time aggregation when appropriate.
- Keep SQL compact.
- For delivery analysis use valid non-null timestamps.
- Never label retained totals as complete revenue.
- Never calculate basket size, item count per order, or average items per order from order_items.

Return only:
## A1 | short title
{tick3}sql
SELECT ...
{tick3}
## A2 | short title
{tick3}sql
SELECT ...
{tick3}
"""
    return parse_plan(ask_llm(prompt),"A",4)

# ---------- SQL repair ----------
def repair_failed_plan(stage_name, question, results):
    tick3 = "`" * 3
    failed = [
        {
            "id": x["id"],
            "title": x["title"],
            "sql": x["sql"],
            "errors": x["messages"],
        }
        for x in results
        if x["status"] in {"BLOCKED", "ERROR"}
    ]

    if not failed:
        return {}

    prompt = f"""
You are repairing SQL for the {stage_name} stage.

USER QUESTION:
{question}

FAILED QUERIES:
{failed}

ACTUAL SQLITE DATABASE SCHEMA:
{SCHEMA}

{SEMANTICS}

Repair every failed query.

STRICT RULES:
- SQLite SELECT/WITH only.
- Every table and column must literally exist in ACTUAL SQLITE DATABASE SCHEMA.
- Do not invent category, segment, order_date, quantity, revenue, or similar generic columns.
- SQLite has no DATE_TRUNC. Use SQLite functions such as strftime when needed.
- Never calculate original basket size, original item count, items per order,
  or original payment-transaction count from transformed tables.
- If the original requested analysis is impossible because its column does not exist,
  replace it with a different useful analysis relevant to the user's question.
- Keep the SAME ID for each repaired query.

Return ONLY markdown blocks like:

## A1 | corrected title
{tick3}sql
SELECT ...
{tick3}

One block for each failed ID. No prose.
"""
    raw = ask_llm(prompt)
    prefix = "A" if stage_name.lower().startswith("analyst") else "T"
    repaired = parse_plan(raw, prefix, len(failed))

    return {item["id"]: item for item in repaired}

def repair_results_once(stage_name, question, results):
    if not any(x["status"] in {"BLOCKED", "ERROR"} for x in results):
        return results

    try:
        repaired_map = repair_failed_plan(stage_name, question, results)
    except Exception:
        return results

    if not repaired_map:
        return results

    new_results = []
    for old_item in results:
        replacement = repaired_map.get(old_item["id"])
        if old_item["status"] in {"BLOCKED", "ERROR"} and replacement:
            repaired_result = execute_plan([replacement])[0]
            new_results.append(repaired_result)
        else:
            new_results.append(old_item)

    return new_results

# ---------- Stage 2 ----------
def paradox_plan(question, primary):
    tick3 = "`" * 3
    evidence = [{"id":x["id"],"title":x["title"],"rows":x["records"][:6]} for x in primary]
    prompt = f"""
You are the PARADOX HUNTER.

USER QUESTION:
{question}

PRIMARY SQL EVIDENCE:
{evidence}

DATABASE:
{SCHEMA}

{SEMANTICS}

Create exactly 2 BROAD, falsifiable SQL tests that could reveal a surprising reversal/tension/subgroup exception.

Rules:
- Do not claim a named state/category/month is paradoxical before seeing the test rows.
- Return comparison tables broad enough for a later Judge to discover WHICH group is surprising.
- Good structures: activity vs delivery performance; rank reversal across two valid metrics;
  time activity vs service outcome; aggregate vs subgroup.
- SELECT/WITH only, SQLite syntax only.
- Use ONLY real tables/columns from DATABASE.
- Avoid LIMIT if it could hide the relevant comparison.
- NEVER calculate basket size, original item count, items per order, or original payment count.
- Payment-method frequency is not customer preference.

Return only:
## T1 | short test question
{tick3}sql
WITH ... SELECT ...
{tick3}
## T2 | short test question
{tick3}sql
SELECT ...
{tick3}
"""
    return parse_plan(ask_llm(prompt),"T",2)

def execute_plan(plan):
    out = []
    for item in plan:
        df, check = execute_sql(item["sql"])
        out.append({
            **item,
            "status":check["status"],
            "messages":check["messages"],
            "records":df.to_dict(orient="records"),
            "columns":df.columns.tolist(),
        })
    return out

# ---------- Stage 3 ----------
def final_report(question, primary, tests):
    p = [{"id":x["id"],"title":x["title"],"rows":x["records"][:8]} for x in primary if x["records"]]
    t = [{"id":x["id"],"title":x["title"],"rows":x["records"][:15]} for x in tests if x["records"]]

    prompt = f"""
You are the FINAL DATA JUDGE + BUSINESS STRATEGIST.

USER QUESTION:
{question}

PRIMARY EVIDENCE:
{p}

PARADOX TEST RESULTS:
{t}

{SEMANTICS}

Tasks:
1. Give 2-4 basic insights from primary evidence.
2. Judge every T-test from actual rows.
3. Report a paradox only if rows directly show a surprising reversal/tension/subgroup exception.
4. Give short/medium/long strategy AND Explicitly forecast the future outcomes (Dự báo).

Strategy rules:
- You MUST distinguish descriptive evidence from business action.
- FORECASTING MANDATE: In SHORT TERM, MEDIUM TERM, and LONG TERM, you MUST format each point into 2 sections: "[Hành động] Đề xuất chiến lược..." và "[Dự báo tương lai] Ước tính cải thiện định lượng dựa trên đà hiện tại...".
- Category retained item-price ranking alone does NOT justify more inventory, advertising, expansion, "high demand", or price changes.
- Payment frequency/value alone does NOT imply customer preference, higher willingness to spend, voucher effectiveness.
- Geographic totals alone do NOT justify expansion or regional marketing.

Return exactly:

## CONCLUSION
short answer

## BASIC INSIGHTS
- ...

## TEST JUDGMENTS
- [T1] SUPPORTED: ...
- [T2] NOT SUPPORTED: ...

## PARADOXICAL INSIGHTS
- only verified paradox, or say none verified

## SHORT TERM
- [Hành động] ...
  [Dự báo tương lai] ...

## MEDIUM TERM
- [Hành động] ...
  [Dự báo tương lai] ...

## LONG TERM
- [Hành động] ...
  [Dự báo tương lai] ...

## LIMITATIONS
- ...

Use the user's language.
"""
    return parse_final(ask_llm(prompt))

# ---------- Resumable workflow cache ----------
def workflow_cache_key(question, history):
    context = "|".join(
        f"{m.get('role','')}:{m.get('content','')}"
        for m in history[-3:]
    )
    raw = f"{question.strip()}||{context}"
    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

def get_workflow_cache():
    if "workflow_cache" not in st.session_state:
        st.session_state.workflow_cache = {}
    return st.session_state.workflow_cache

def cached_stage_state(key):
    cache = get_workflow_cache()
    if key not in cache:
        cache[key] = {
            "primary": None,
            "usable_primary": None,
            "tests": None,
            "usable_tests": None,
        }
    return cache[key]

def clear_workflow_cache_for_key(key):
    cache = get_workflow_cache()
    cache.pop(key, None)

# ---------- Workflow ----------
def empty_strategy():
    return {"short_term":[],"medium_term":[],"long_term":[]}

def err_result(message, trace, primary=None, tests=None):
    return {
        "answer":message,"basic_insights":[],"paradoxical_insights":[],
        "strategy":empty_strategy(),"limitations":[],"judgments":[],
        "primary_analyses":primary or [],"paradox_candidates":tests or [],
        "stage_trace":trace,"chart":{"type":"none"},"error":True,
    }

def ask_agent(question, history):
    q = question.lower()
    
    trace = {
        "analyst": "PENDING",
        "paradox_hunter": "PENDING",
        "paradox_verification": "PENDING",
        "strategist": "PENDING",
    }

    key = workflow_cache_key(question, history)
    state = cached_stage_state(key)

    hist = "\n".join(
        f"{m['role']}: {m.get('content','')}"
        for m in history[-3:]
    )

    # Stage 1 — Analyst
    if state["usable_primary"] is not None:
        primary = state["primary"]
        usable = state["usable_primary"]
        trace["analyst"] = f"CACHED — reused {len(usable)} analysis table(s)"
    else:
        trace["analyst"] = "RUNNING"
        try:
            plan = analyst_plan(question, hist)
        except Exception as e:
            trace["analyst"] = "FAILED"
            return err_result(friendly_error(e, "Analyst"), trace)

        if not plan:
            trace["analyst"] = "FAILED — no SQL blocks parsed"
            return err_result("Analyst did not return executable SQL blocks. Retry once.", trace)

        primary = execute_plan(plan)
        primary = repair_results_once("Analyst", question, primary)

        usable = [
            x for x in primary
            if (x["status"] in {"SAFE", "WARNING"} and x["records"])
        ]

        if not usable:
            trace["analyst"] = "FAILED — no usable SQL evidence"
            return err_result("Analyst generated no usable SQL evidence.", trace, primary)

        broad_request = any(
            token in q for token in ["insight", "business", "doanh nghiệp", "chiến lược", "strategy"]
        )

        if broad_request and len(usable) < 2:
            trace["analyst"] = f"INSUFFICIENT — only {len(usable)} usable analysis table(s)"
            return err_result(
                "Chỉ có một chiều phân tích hợp lệ sau khi kiểm tra SQL, nên agent "
                "không tạo broad business strategy để tránh suy diễn quá mức. "
                "Hãy thử lại; app sẽ yêu cầu Analyst dùng schema thật để tạo thêm evidence.",
                trace, primary
            )

        state["primary"] = primary
        state["usable_primary"] = usable
        trace["analyst"] = f"COMPLETED — {len(usable)} table(s)"

    # Stage 2 — Paradox Hunter
    if state["usable_tests"] is not None:
        tests = state["tests"]
        usable_tests = state["usable_tests"]
        trace["paradox_hunter"] = f"CACHED — reused {len(tests)} test(s)"
        trace["paradox_verification"] = f"CACHED — reused {len(usable_tests)} test table(s)"
    else:
        trace["paradox_hunter"] = "RUNNING"
        try:
            test_plan = paradox_plan(question, usable)
        except Exception as e:
            if is_rate_limit(e):
                trace["paradox_hunter"] = "RATE LIMITED"
                trace["paradox_verification"] = "NOT RUN"
                trace["strategist"] = "NOT RUN"
                return err_result(
                    friendly_error(e, "Paradox Hunter")
                    + "\n\n**Progress saved:** Analyst SQL evidence has been cached. "
                      "After the wait time, send the same question again and the app "
                      "will resume from Paradox Hunter instead of rerunning Analyst.",
                    trace, primary=primary
                )
            test_plan = []
            trace["paradox_hunter"] = "FAILED — " + friendly_error(e, "Paradox Hunter")

        tests = execute_plan(test_plan) if test_plan else []
        if tests:
            tests = repair_results_once("Paradox Hunter", question, tests)

        usable_tests = [
            x for x in tests
            if (x["status"] in {"SAFE", "WARNING"} and x["records"])
        ]

        state["tests"] = tests
        state["usable_tests"] = usable_tests

        if test_plan:
            trace["paradox_hunter"] = f"COMPLETED — {len(test_plan)} test(s)"
            trace["paradox_verification"] = f"COMPLETED — {len(usable_tests)} test table(s)"
        else:
            trace["paradox_verification"] = "SKIPPED — no executable test"

    # Stage 3 — Final Judge + Strategist
    trace["strategist"] = "RUNNING"
    try:
        report = final_report(question, usable, usable_tests)
    except Exception as e:
        if is_rate_limit(e):
            trace["strategist"] = "RATE LIMITED"
            return err_result(
                friendly_error(e, "Final Judge + Strategist")
                + "\n\n**Progress saved:** Analyst evidence and Paradox SQL tests "
                  "have been cached. After the wait time, send the same question again; "
                  "the app will resume directly at the Final Judge.",
                trace, primary=primary, tests=tests
            )
        trace["strategist"] = "FAILED"
        return err_result(friendly_error(e, "Final Judge + Strategist"), trace, primary, tests)

    trace["strategist"] = "COMPLETED"

    supported = sum(1 for j in report["judgments"] if j["supported"])
    trace["paradox_verification"] += f" | {supported} verified"

    # Default logic for interactive chart trigger
    chart = {"type": "none"}
    if any(w in q for w in ["chart", "graph", "plot", "visual", "visualization", "biểu đồ", "vẽ"]):
        for x in usable:
            df = pd.DataFrame(x["records"])
            if df.empty: continue
            nums = df.select_dtypes(include="number").columns.tolist()
            cats = [c for c in df.columns if c not in nums]
            if nums:
                chart = {
                    "type": "bar",
                    "x": cats[0] if cats else nums[0],
                    "y": nums[0] if (len(nums)==1 or not cats) else (nums[1] if nums[0]==cats[0] else nums[0]),
                    "title": x["title"],
                    "source_id": x["id"],
                }
                break

    clear_workflow_cache_for_key(key)

    return {
        **report,
        "primary_analyses": primary,
        "paradox_candidates": tests,
        "stage_trace": trace,
        "chart": chart,
        "error": False,
    }

# ---------- Multi-chat & UI ----------
def create_chat():
    cid = str(uuid.uuid4())
    st.session_state.chats[cid] = {"title": "New chat", "messages": []}
    st.session_state.current_chat_id = cid

def delete_chat(cid):
    if cid in st.session_state.chats:
        del st.session_state.chats[cid]
    if not st.session_state.chats:
        create_chat()
    elif st.session_state.current_chat_id == cid:
        st.session_state.current_chat_id = next(reversed(st.session_state.chats))

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
    create_chat()

# ========================================================
# ---------- UI: CẤU HÌNH & SIDEBAR ----------
# ========================================================
with st.sidebar:
    st.markdown("### 🔥 Group 3 - TINE313")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Chat Mới", type="primary", use_container_width=True):
            create_chat()
            st.rerun()
    with col2:
        with st.popover("⚙️ Cấu hình", use_container_width=True):
            st.text_input("Gemini API Key:", type="password", key="api_key")
            st.markdown("[👉 Lấy API Key tại đây](https://aistudio.google.com/app/apikey)")
            st.markdown("---")
            st.markdown("**Kết nối MySQL (Tùy chọn)**")
            st.caption("Để trống nếu dùng CSDL SQLite mặc định.")
            st.text_input("Host (VD: localhost):", key="db_host")
            st.text_input("Port (VD: 3306):", key="db_port")
            st.text_input("Username (VD: root):", key="db_user")
            st.text_input("Password:", type="password", key="db_pass")
            st.text_input("Database Name:", key="db_name")

    st.markdown("---")
    st.markdown("💬 **Chats**")

    for cid, chat in reversed(list(st.session_state.chats.items())):
        a, b = st.columns([.82, .18])
        current = (cid == st.session_state.current_chat_id)
        with a:
            if st.button(("● " if current else "") + chat["title"], key=f"open_{cid}", use_container_width=True):
                st.session_state.current_chat_id = cid
                st.rerun()
        with b:
            if st.button("🗑️", key=f"del_{cid}", use_container_width=True):
                delete_chat(cid)
                st.rerun()

current = st.session_state.chats[st.session_state.current_chat_id]
history = current["messages"]

# ========================================================
# ---------- UI: MAIN PAGE ----------
# ========================================================
st.title("🛒 My AI agent")
st.markdown("Trợ lý AI phân tích dữ liệu, săn Insight & Hoạch định Chiến lược")
st.markdown("🔥 **Agent phát triển bởi: Group 3 - TINE313** 🔥")
st.caption("Analyst → Paradox Hunter → SQL Verification → Final Judge & Strategy • Read-only SQLite")

def render_list(items, empty):
    vals = [str(x).strip() for x in (items or []) if str(x).strip()]
    if not vals:
        st.info(empty)
        return
    for x in vals:
        st.markdown(f"- {x}")

def render_audit(r):
    t = r.get("stage_trace", {})
    with st.expander("🔍 Auto-Audit & Agent Workflow"):
        st.markdown(f"- **1. Analyst:** {t.get('analyst','UNKNOWN')}")
        st.markdown(f"- **2. Paradox Hunter:** {t.get('paradox_hunter','UNKNOWN')}")
        st.markdown(f"- **3. SQL Verification:** {t.get('paradox_verification','UNKNOWN')}")
        st.markdown(f"- **4. Final Judge + Strategist:** {t.get('strategist','UNKNOWN')}")

def render_chart(r, msg_hash):
    c = r.get("chart", {})
    if c.get("type") == "none": return
    
    for x in r.get("primary_analyses", []):
        if x["id"] == c.get("source_id"):
            df = pd.DataFrame(x["records"])
            if df.empty: continue
            
            st.markdown("---")
            st.markdown("### 📊 Interactive Dashboard (Power BI Style)")
            
            w_key = f"{msg_hash}_{x['id']}"
            all_cols = df.columns.tolist()
            
            def_x = c.get("x") if c.get("x") in all_cols else all_cols[0]
            def_y = c.get("y") if c.get("y") in all_cols else all_cols[-1]
            
            viz_col, ctrl_col = st.columns([3, 1])
            
            with ctrl_col:
                st.markdown("⚙️ **Visualizations Pane**")
                c_types = ["bar", "line", "pie", "scatter", "area"]
                sel_type = st.selectbox("Loại biểu đồ", c_types, index=0, key=f"type_{w_key}")
                sel_x = st.selectbox("Trục X (Dimension)", all_cols, index=all_cols.index(def_x), key=f"x_{w_key}")
                sel_y = st.selectbox("Trục Y (Measure)", all_cols, index=all_cols.index(def_y), key=f"y_{w_key}")
                color_choice = st.selectbox("Phân loại (Legend)", ["None"] + all_cols, key=f"color_{w_key}")
                show_trend = st.checkbox("📈 Bật đường dự báo (Trendline)", key=f"trend_{w_key}")
                
            with viz_col:
                try:
                    color_arg = None if color_choice == "None" else color_choice
                    title = c.get("title", "Data Visualization")
                    
                    if sel_type == "scatter":
                        if show_trend:
                            try:
                                fig = px.scatter(df, x=sel_x, y=sel_y, color=color_arg, trendline="ols", title=title)
                            except Exception:
                                fig = px.scatter(df, x=sel_x, y=sel_y, color=color_arg, title=title)
                                st.warning("⚠️ Cần cài đặt thư viện 'statsmodels' (pip install statsmodels) để vẽ đường xu hướng OLS.")
                        else:
                            fig = px.scatter(df, x=sel_x, y=sel_y, color=color_arg, title=title)
                    elif sel_type == "bar":
                        fig = px.bar(df, x=sel_x, y=sel_y, color=color_arg, title=title)
                    elif sel_type == "line":
                        fig = px.line(df, x=sel_x, y=sel_y, color=color_arg, title=title)
                    elif sel_type == "pie":
                        fig = px.pie(df, names=sel_x, values=sel_y, title=title)
                    elif sel_type == "area":
                        fig = px.area(df, x=sel_x, y=sel_y, color=color_arg, title=title)
                        
                    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Dữ liệu không tương thích. Vui lòng đổi trục X/Y. (Lỗi: {e})")

def render_evidence(r):
    st.markdown("### Primary evidence")
    for x in r.get("primary_analyses", []):
        st.markdown(f"#### {x['id']} — {x['title']}")
        st.caption(f"SQL validation: {x['status']}")
        df = pd.DataFrame(x["records"])
        if not df.empty: st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("---")
    st.markdown("### Paradox tests")
    jm = {j["id"]: j for j in r.get("judgments", [])}
    tests = r.get("paradox_candidates", [])
    if not tests: st.info("No executable paradox test was produced in this run.")
    for x in tests:
        st.markdown(f"#### {x['id']} — {x['title']}")
        df = pd.DataFrame(x["records"])
        if not df.empty: st.dataframe(df, use_container_width=True, hide_index=True)
        j = jm.get(x["id"])
        if j:
            (st.success if j["supported"] else st.warning)(("SUPPORTED — " if j["supported"] else "NOT SUPPORTED — ") + j["reason"])

def render_sql(r):
    st.markdown("### Analyst SQL")
    for x in r.get("primary_analyses", []):
        st.markdown(f"#### {x['id']} — {x['status']}")
        st.code(x["sql"], language="sql")
        render_list(x["messages"], "")
    st.markdown("---")
    st.markdown("### Paradox verification SQL")
    for x in r.get("paradox_candidates", []):
        st.markdown(f"#### {x['id']} — {x['status']}")
        st.code(x["sql"], language="sql")
        render_list(x["messages"], "")

def render_report(r, msg_hash):
    if r.get("error"):
        st.error(r["answer"])
        render_audit(r)
        return
    st.success("💡 AI Agent đã hoàn tất: phân tích → tìm nghịch lý → kiểm chứng SQL → chiến lược.")
    render_audit(r)
    a, b, c, d = st.tabs(["📊 Báo cáo Insight", "💡 Chiến lược & Dự báo", "🧪 Evidence & Paradox Test", "⚙️ SQL"])
    with a:
        st.markdown("### Kết luận")
        st.markdown(r["answer"])
        render_chart(r, msg_hash)
        st.markdown("### 1. Insight cơ bản")
        render_list(r.get("basic_insights", []), "No basic insight produced.")
        st.markdown("### 2. Insight nghịch lý")
        render_list(r.get("paradoxical_insights", []), "No verified paradox.")
        if r.get("limitations"):
            with st.expander("⚠️ Giới hạn diễn giải"):
                render_list(r["limitations"], "")
    with b:
        s = r.get("strategy", empty_strategy())
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### ⚡ Ngắn hạn")
            render_list(s.get("short_term", []), "No short-term recommendation.")
        with c2:
            st.markdown("### 🧭 Trung hạn")
            render_list(s.get("medium_term", []), "No medium-term recommendation.")
        with c3:
            st.markdown("### 🏗️ Dài hạn")
            render_list(s.get("long_term", []), "No long-term recommendation.")
    with c:
        render_evidence(r)
    with d:
        render_sql(r)

def render_message(m, idx):
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        elif m.get("result"):
            msg_hash = f"chat_{st.session_state.current_chat_id}_msg_{idx}"
            render_report(m["result"], msg_hash)
        else:
            st.markdown(m.get("content", ""))

current = st.session_state.chats[st.session_state.current_chat_id]
history = current["messages"]

for idx, m in enumerate(history):
    render_message(m, idx)

question = st.chat_input("VD: Cho tôi insights về doanh thu và vẽ biểu đồ...")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.spinner("Agent đang phân tích và dự báo..."):
        result = ask_agent(question, history)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result["answer"], "result": result})

    if current["title"] == "New chat":
        title = " ".join(question.split())
        current["title"] = title[:34] + ("..." if len(title) > 34 else "")

    st.rerun()
