import streamlit as st
import pandas as pd
import json
import os
import uuid
import plotly.express as px
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from sqlalchemy import create_engine

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 1. ĐỌC CODEBOOK (TỪ ĐIỂN DỮ LIỆU)
# ==========================================
CODEBOOK_PATH = Path("ecommerce_agent_codebook.md")
try:
    CODEBOOK_TEXT = CODEBOOK_PATH.read_text(encoding="utf-8")
except Exception:
    CODEBOOK_TEXT = "Không tìm thấy file Codebook."

# ==========================================
# 2. QUẢN LÝ LỊCH SỬ HỘI THOẠI
# ==========================================
HISTORY_FILE = "chat_history_v2.json"
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
def save_history(all_chats):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chats, f, ensure_ascii=False, indent=4)

# ==========================================
# 3. KHUÔN ĐÚC ĐẦU RA (TƯ DUY V1 + AN TOÀN V2)
# ==========================================
class ChartSpec(BaseModel):
    type: Literal["none", "bar", "line", "scatter", "pie"] = Field(description="Loại biểu đồ. Chọn 'none' nếu không cần.")
    x: Optional[str] = Field(description="Tên cột trục X (BẮT BUỘC phải tồn tại trong kết quả SQL)")
    y: Optional[str] = Field(description="Tên cột trục Y (BẮT BUỘC phải tồn tại trong kết quả SQL)")
    title: Optional[str] = Field(description="Tiêu đề biểu đồ")

class Presentation(BaseModel):
    answer: str = Field(description="Câu trả lời giao tiếp tự nhiên, thân thiện với người dùng.")
    sql_query: str = Field(description="Câu lệnh MySQL hợp lệ. LUÔN dùng DISTINCT khi đếm ID.")
    basic_insights: List[str] = Field(description="Insight cơ bản: Đọc vị các con số tổng quan, xu hướng chính.")
    deep_insights: List[str] = Field(description="Insight chuyên sâu/Nghịch lý: Phát hiện điểm bất thường, rủi ro ngầm, hoặc cơ hội ẩn giấu.")
    short_term_strategy: List[str] = Field(description="Chiến lược Ngắn hạn (Cấp bách) dựa trên dữ liệu.")
    medium_term_strategy: List[str] = Field(description="Chiến lược Trung hạn dựa trên dữ liệu.")
    long_term_strategy: List[str] = Field(description="Chiến lược Dài hạn dựa trên dữ liệu.")
    chart: ChartSpec = Field(description="Cấu hình biểu đồ Plotly minh họa cho Insight.")

# ==========================================
# 4. BỘ NÃO XỬ LÝ (AGENT - GEMINI 3.6 FLASH & MYSQL)
# ==========================================
SYSTEM_PROMPT = f"""Bạn là Giám đốc Vận hành (COO) & Kỹ sư Dữ liệu cấp cao tại một E-commerce Marketplace.

ĐÂY LÀ TỪ ĐIỂN DỮ LIỆU CỦA HỆ THỐNG (CODEBOOK):
{CODEBOOK_TEXT}

CƠ SỞ DỮ LIỆU: Hệ thống sử dụng CSDL MySQL. Mọi câu lệnh truy vấn phải tuân thủ nghiêm ngặt cú pháp của MySQL.

QUY TẮC BẮT BUỘC:
1. SỰ THẬT DỮ LIỆU: LUÔN viết SQL để lấy số liệu thực. Không tự bịa số liệu. Tuân thủ định nghĩa doanh thu trong Codebook.
2. TƯ DUY PHÂN TÍCH: Phải luôn tìm ra các nghịch lý hoặc rủi ro ngầm (Deep Insights) chứ không chỉ đọc số liệu bề nổi.
3. CHIẾN LƯỢC: Các đề xuất chiến lược (Ngắn, Trung, Dài hạn) phải bám sát vào những con số vừa tìm được.
4. BIỂU ĐỒ: Cấu hình ChartSpec hợp lý. Tên cột x, y phải khớp 100% với tên cột bạn SELECT trong câu SQL.
"""

def analyze_data(question, api_key):
    # Sử dụng đúng phiên bản Gemini 3.6 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, temperature=0.1)
    structured_llm = llm.with_structured_output(Presentation)
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]
    
    result = structured_llm.invoke(messages)
    
    data = []
    if result.sql_query:
        try:
            if any(kw in result.sql_query.upper() for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER"]):
                raise Exception("Phát hiện mã SQL thay đổi dữ liệu bị cấm!")
                
            # Kết nối MySQL bằng thông tin bảo mật
            db_user = st.secrets["DB_USER"]
            db_pass = st.secrets["DB_PASSWORD"]
            db_host = st.secrets["DB_HOST"]
            db_port = st.secrets.get("DB_PORT", 3306)
            db_name = st.secrets["DB_NAME"]
            
            engine = create_engine(f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")
            df = pd.read_sql_query(result.sql_query, engine)
            data = df.to_dict(orient="records")
            
        except Exception as e:
            result.answer += f"\n\n(⚠️ Lỗi SQL / MySQL: {e})"
            
    return result, data

# ==========================================
# 5. GIAO DIỆN STREAMLIT CHUẨN UX/UI V1
# ==========================================
st.set_page_config(page_title="My AI agent", page_icon="🛒", layout="wide")
st.title("🛒 My AI agent")
st.markdown("Trợ lý AI phân tích dữ liệu, săn Insight & Hoạch định Chiến lược")
st.markdown("🔥 **Agent phát triển bởi: Group 3 - TINE313** 🔥")

if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_history()
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
    if st.session_state.current_session_id not in st.session_state.all_chats:
        st.session_state.all_chats[st.session_state.current_session_id] = []

with st.sidebar:
    st.markdown("### 🔥 Group 3 - TINE313")
    st.markdown("---")
    
    if st.button("➕ Chat Mới", type="primary", use_container_width=True):
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.all_chats[st.session_state.current_session_id] = []
        st.rerun()

    st.markdown("---")
    st.markdown("📂 **Danh mục Bảng Dữ liệu (MySQL)**")
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
    for session_id, chat_messages in reversed(st.session_state.all_chats.items()):
        if len(chat_messages) > 0:
            has_history = True
            title = "Tin nhắn mới..."
            for m in chat_messages:
                if m["role"] == "user":
                    title = m["content"][:22] + "..."
                    break
            is_active = (session_id == st.session_state.current_session_id)
            btn_label = f"👉 {title}" if is_active else f"💬 {title}"
            if st.button(btn_label, key=f"hist_{session_id}", use_container_width=True):
                st.session_state.current_session_id = session_id
                st.rerun()
    if not has_history:
        st.info("Chưa có lịch sử trò chuyện.")

for msg in st.session_state.all_chats[st.session_state.current_session_id]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("VD: Cho tôi insight về doanh thu theo từng tiểu bang..."):
    # Tự động lấy API Key từ két sắt (Secrets) của Streamlit
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("⚠️ Lỗi: Chưa cấu hình GEMINI_API_KEY trong mục Settings > Secrets của Streamlit Cloud!")
        st.stop()

    st.session_state.all_chats[st.session_state.current_session_id].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner(f"Agent đang xử lý bằng gemini-3.6-flash & truy vấn MySQL..."):
            try:
                result_obj, data = analyze_data(prompt, api_key)
                
                st.markdown(result_obj.answer)
                st.markdown("💡 **Hệ thống AI đã bóc tách thành công các Insight chuyên sâu từ CSDL.**")
                
                tab1, tab2, tab3 = st.tabs(["📊 Insight & Biểu đồ", "💡 Chiến lược", "⚙️ Dữ liệu & SQL"])
                
                with tab1:
                    st.markdown("### 1. Insight Cơ bản")
                    for ins in result_obj.basic_insights:
                        st.write(f"🔹 {ins}")
                        
                    st.markdown("### 2. Insight Chuyên sâu & Nghịch lý")
                    for ins in result_obj.deep_insights:
                        st.write(f"⚠️ **{ins}**")
                    
                    if data and result_obj.chart.type != "none":
                        st.markdown("---")
                        df_chart = pd.DataFrame(data)
                        c = result_obj.chart
                        try:
                            if c.type == "bar":
                                st.plotly_chart(px.bar(df_chart, x=c.x, y=c.y, title=c.title), use_container_width=True)
                            elif c.type == "pie":
                                st.plotly_chart(px.pie(df_chart, names=c.x, values=c.y, title=c.title), use_container_width=True)
                            elif c.type == "line":
                                st.plotly_chart(px.line(df_chart, x=c.x, y=c.y, title=c.title), use_container_width=True)
                            elif c.type == "scatter":
                                st.plotly_chart(px.scatter(df_chart, x=c.x, y=c.y, title=c.title), use_container_width=True)
                        except Exception as e:
                            st.warning(f"Cấu trúc biểu đồ AI đề xuất chưa khớp với dữ liệu: {e}")
                            
                with tab2:
                    st.markdown("### 🚀 Chiến lược Ngắn hạn (Cấp bách)")
                    for strat in result_obj.short_term_strategy:
                        st.write(f"⚡ {strat}")
                        
                    st.markdown("### 📈 Chiến lược Trung hạn")
                    for strat in result_obj.medium_term_strategy:
                        st.write(f"🎯 {strat}")
                        
                    st.markdown("### 🌍 Chiến lược Dài hạn")
                    for strat in result_obj.long_term_strategy:
                        st.write(f"🌟 {strat}")
                        
                with tab3:
                    st.markdown("**Câu lệnh SQL đã thực thi:**")
                    st.code(result_obj.sql_query, language="sql")
                    if data:
                        st.markdown("**🗄️ Bảng kết quả (Data Preview):**")
                        st.dataframe(pd.DataFrame(data), use_container_width=True)
                
                st.session_state.all_chats[st.session_state.current_session_id].append(
                    {"role": "assistant", "content": result_obj.answer + "\n\n*(Xem chi tiết Insight, Chiến lược và Biểu đồ tại các Tab)*"}
                )
                save_history(st.session_state.all_chats)
                
            except Exception as e:
                st.error(f"Lỗi hệ thống: {e}")