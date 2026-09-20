import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
import re
import pandas as pd
from sqlalchemy import create_engine
import json
import os
import uuid
import sqlite3
import textwrap

# ==========================================
# 0. KHỞI TẠO ĐƯỜNG DẪN & DATABASE SQLITE (ĐỌC CSV BẤT TỬ)
# ==========================================
DB_PATH = os.path.abspath("ecommerce.db").replace('\\', '/')
DB_URI_SQLITE = f"sqlite:///{DB_PATH}"

@st.cache_resource(show_spinner="Đang nạp 89.000+ đơn hàng từ CSV vào hệ thống... Vui lòng đợi vài giây!")
def init_sqlite_db():
    conn = sqlite3.connect("ecommerce.db", check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='df_orders'")
    has_table = cursor.fetchone()[0] > 0
    
    needs_update = True
    if has_table:
        cursor.execute("SELECT COUNT(*) FROM df_orders")
        row_count = cursor.fetchone()[0]
        if row_count > 100:
            needs_update = False
            
    if needs_update:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        read_opts = {'sep': None, 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'utf-8'}
        
        tables = {
            'df_customers': ['df_Customers.csv', 'df_customers.csv'],
            'df_orders': ['df_Orders.csv', 'df_orders.csv'],
            'df_payments': ['df_Payments.csv', 'df_payments.csv'],
            'df_products': ['df_Products.csv', 'df_products.csv'],
            'df_orderitems': ['df_OrderItems.csv', 'df_orderitems.csv']
        }
        
        for table_name, file_names in tables.items():
            file_found = False
            for f_name in file_names:
                full_path = os.path.join(BASE_DIR, f_name)
                if os.path.exists(full_path):
                    try:
                        # Dọn dẹp duplicate ngay khi đọc
                        df = pd.read_csv(full_path, **read_opts).drop_duplicates()
                        df.to_sql(table_name, conn, index=False, if_exists='replace')
                        file_found = True
                        break
                    except Exception as e:
                        st.error(f"Lỗi khi nạp dữ liệu {f_name}: {e}")
                        break
            
            if not file_found:
                st.error(f"❌ Không tìm thấy file {file_names[0]}. Hãy chắc chắn file đang nằm chung thư mục với agent_app.py tại: {BASE_DIR}")
                
    conn.close()
    return True

init_sqlite_db()

# ==========================================
# 0. QUẢN LÝ LỊCH SỬ HỘI THOẠI (MULTI-SESSION)
# ==========================================
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    if len(data) > 0:
                        return {str(uuid.uuid4()): data}
                    return {}
                return data
        except:
            return {}
    return {}

def save_history(all_chats):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chats, f, ensure_ascii=False, indent=4)

def clear_all_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.session_state.all_chats = {}
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.all_chats[st.session_state.current_session_id] = []

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="My AI agent", page_icon="🛒", layout="wide")
st.title("🛒 My AI agent")
st.markdown("Trợ lý AI phân tích dữ liệu, săn Insight & Hoạch định Chiến lược")
st.markdown("🔥 **Agent phát triển bởi: Group 3 - TINE313** 🔥")

if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_history()

if "current_session_id" not in st.session_state:
    if st.session_state.all_chats:
        st.session_state.current_session_id = list(st.session_state.all_chats.keys())[-1]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.all_chats[new_id] = []

# ==========================================
# 2. KHU VỰC CẤU HÌNH & SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 🔥 Group 3 - TINE313")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Chat Mới", type="primary", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.current_session_id = new_id
            st.session_state.all_chats[new_id] = []
            st.rerun()
    with col2:
        with st.popover("⚙️ Cấu hình"):
            google_api_key = st.text_input("Gemini API Key:", type="password", key="api_key")
            st.markdown("[👉 Lấy API Key tại đây](https://aistudio.google.com/app/apikey)")
            st.markdown("---")
            st.caption("Để trống MySQL nếu muốn dùng file ecommerce.db")
            mysql_host = st.text_input("MySQL Host:", key="db_host")
            mysql_user = st.text_input("Username:", key="db_user")
            mysql_pass = st.text_input("Password:", type="password", key="db_pass")
            mysql_db = st.text_input("Database:", key="db_name")

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

    if st.button("🗑️ Dọn dẹp TOÀN BỘ lịch sử", use_container_width=True):
        clear_all_history()
        st.rerun()

# ==========================================
# 3. BỘ NÃO CHIẾN LƯỢC & ÉP KHUÔN ĐẦU RA
# ==========================================
instructions_raw = """
# VAI TRÒ
Bạn là Giám đốc Vận hành (COO) & Kỹ sư Dữ liệu cấp cao tại một E-commerce Marketplace.

# QUY TRÌNH VẬN HÀNH BẮT BUỘC (SOP)
1. TÌM KIẾM SỰ THẬT: BẠN BẮT BUỘC phải dùng công cụ sql_db_query để truy vấn CSDL. TUYỆT ĐỐI KHÔNG tự bịa số liệu.
2. NỐI BẢNG: Luôn dùng df_orders làm trung tâm. Tính doanh thu bằng SUM(payment_value), loại trừ đơn Cancelled.
3. KHỬ TRÙNG LẶP: Dùng từ khóa DISTINCT (ví dụ: COUNT(DISTINCT order_id)) để đảm bảo số liệu không bị x2, x3.

# ĐỊNH DẠNG ĐẦU RA BẮT BUỘC (FINAL ANSWER):
Khi bạn đã có kết quả cuối cùng, bạn BẮT BUỘC phải bắt đầu bằng cụm từ "Final Answer: " sau đó mới đến các thẻ.

Final Answer:
[BIỂU ĐỒ]
(BẮT BUỘC CHỦ ĐỘNG VẼ BIỂU ĐỒ MINH HỌA CHO INSIGHT.
⚠️ LUẬT VẼ BIỂU ĐỒ: BẠN PHẢI TỰ HARDCODE DỮ LIỆU ĐÃ TRUY VẤN ĐƯỢC VÀO PANDAS DATAFRAME TRONG ĐOẠN CODE NÀY. TUYỆT ĐỐI KHÔNG CHÈN THÊM BẤT KỲ CÂU TIẾNG ANH NÀO VÀO TRONG CODE.
Gộp toàn bộ code streamlit (st.bar_chart, st.line_chart...) vào DUY NHẤT 1 khối '''python)
'''python
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
# Tạo dataframe từ số liệu thô và vẽ biểu đồ tại đây
'''

[PHÂN TÍCH]
(Trình bày phân tích bằng Markdown sắc nét, chia làm 2 ý rõ ràng: Insight cơ bản và Insight chuyên sâu)

[CHIẾN LƯỢC]
(Trình bày bằng Bullet Points: Ngắn hạn, Trung hạn, Dài hạn)

[SQL]
'''sql
-- Dán câu lệnh SQL đã chạy thành công
'''
"""
tick3 = chr(96) * 3
instructions = instructions_raw.replace("'''", tick3)

# ==========================================
# 4. WORKFLOW KIỂM ĐỊNH & KẾT NỐI DATABASE
# ==========================================
def get_db_uri():
    host = st.session_state.get("db_host", "")
    user = st.session_state.get("db_user", "")
    pwd = st.session_state.get("db_pass", "")
    db_name = st.session_state.get("db_name", "")
    
    if host and user and db_name:
        pwd_part = f":{pwd}" if pwd else ""
        return f"mysql+pymysql://{user}{pwd_part}@{host}:3306/{db_name}"
    return DB_URI_SQLITE

def run_data_audit(db_uri):
    engine = create_engine(db_uri)
    audit_logs = []
    try:
        df_pay = pd.read_sql("SELECT order_id, payment_value FROM df_payments WHERE payment_value < 0", engine)
        if not df_pay.empty:
            audit_logs.append({"status": "error", "msg": f"❌ df_payments: Phát hiện {len(df_pay)} giao dịch có giá trị âm."})
        else:
            audit_logs.append({"status": "success", "msg": "✅ df_payments: 100% giao dịch hợp lệ."})
            
        df_ord = pd.read_sql("SELECT order_id FROM df_orders WHERE order_status IS NULL OR order_status = ''", engine)
        if not df_ord.empty:
            audit_logs.append({"status": "warning", "msg": f"⚠️ df_orders: Phát hiện {len(df_ord)} đơn hàng bị trống trạng thái."})
        else:
            audit_logs.append({"status": "success", "msg": "✅ df_orders: Toàn vẹn dữ liệu."})
            
        df_dup_ord = pd.read_sql("SELECT order_id FROM df_orders GROUP BY order_id HAVING COUNT(order_id) > 1", engine)
        df_dup_cus = pd.read_sql("SELECT customer_id FROM df_customers GROUP BY customer_id HAVING COUNT(customer_id) > 1", engine)
        total_dups = len(df_dup_ord) + len(df_dup_cus)
        
        if total_dups > 0:
            audit_logs.append({"status": "warning", "msg": f"⚠️ Cảnh báo rác dữ liệu: Phát hiện {total_dups} ID bị nhân bản dòng (Duplicates). Kích hoạt lệnh ép AI dùng DISTINCT."})
        else:
            audit_logs.append({"status": "success", "msg": "✅ Dữ liệu định danh: Sạch sẽ, không phát hiện lỗi nhân bản dòng."})
            
    except Exception as e:
        audit_logs.append({"status": "warning", "msg": f"⚠️ Bỏ qua kiểm định sâu do CSDL chưa khởi tạo đầy đủ."})
        
    return audit_logs

# ==========================================
# 5. KHỞI TẠO TÁC NHÂN
# ==========================================
def get_agent():
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        return None, "Vui lòng nhập API Key trong mục Cấu hình."
    try:
        db_uri = get_db_uri()
        db = SQLDatabase.from_uri(db_uri)
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, temperature=0.1)
        
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=None,
            db=db,
            agent_type="zero-shot-react-description",
            prefix=instructions,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15
        )
        return agent_executor, "OK"
    except Exception as e:
        return None, str(e)

# ==========================================
# 6. GIAO DIỆN HIỂN THỊ (BẢN CẤP CỨU CHỐNG LỖI)
# ==========================================
def render_assistant_response(answer, audit_logs=None):
    answer = answer.replace("`", "") if answer.startswith("`") else answer
    
    # --- TRÍCH XUẤT CODE PYTHON & SQL CỰC AN TOÀN ---
    code_blocks = []
    raw_py_blocks = re.findall(fr'{tick3}python\s*(.*?){tick3}', answer, re.DOTALL | re.IGNORECASE)
    for b in raw_py_blocks:
        code_blocks.append(b)

    sql_blocks = []
    raw_sql_blocks = re.findall(fr'{tick3}sql\s*(.*?){tick3}', answer, re.DOTALL | re.IGNORECASE)
    for s in raw_sql_blocks:
        s_clean = s.strip()
        if s_clean and s_clean not in sql_blocks:
            sql_blocks.append(s_clean)
            
    # --- THUẬT TOÁN BÓC TÁCH VĂN BẢN THÔNG MINH ---
    phan_tich = answer
    chien_luoc = "Hệ thống chưa kịp hoàn thiện chiến lược hoặc chiến lược đã được gộp chung ở Tab Báo cáo Phân tích."
    
    if "[PHÂN TÍCH]" in answer:
        try:
            phan_tich = answer.split("[PHÂN TÍCH]")[-1].split("[CHIẾN LƯỢC]")[0].split("[SQL]")[0].strip()
        except:
            pass
            
    if "[CHIẾN LƯỢC]" in answer:
        try:
            chien_luoc = answer.split("[CHIẾN LƯỢC]")[-1].split("[SQL]")[0].split("[PHÂN TÍCH]")[0].strip()
        except:
            pass
            
    if phan_tich == answer:
        phan_tich = re.sub(fr'{tick3}.*?{tick3}', '', phan_tich, flags=re.DOTALL)
        phan_tich = phan_tich.replace("Final Answer:", "").replace("[BIỂU ĐỒ]", "").strip()

    # --- THỰC THI BIỂU ĐỒ (LỌC SẠCH RÁC TIẾNG ANH TRONG CODE) ---
    if code_blocks:
        combined_code = "\n".join(code_blocks)
        
        # LỌC RÁC: Xóa ngay lập tức mọi dòng bắt đầu bằng dấu "..." hoặc có chứa chữ "EXACTLY ONE python code block"
        clean_lines = []
        for line in combined_code.split('\n'):
            if not line.strip().startswith('...') and "EXACTLY ONE python code block" not in line:
                clean_lines.append(line)
        
        clean_code = "\n".join(clean_lines)
        clean_code = textwrap.dedent(clean_code).strip()
        
        if clean_code:
            try:
                # Import đầy đủ thư viện để phòng ngừa lỗi
                exec_globals = {'st': st, 'pd': pd, 'plt': __import__('matplotlib.pyplot').pyplot, 'sns': __import__('seaborn')}
                exec(clean_code, exec_globals)
            except Exception as e:
                st.warning(f"⚠️ **AI viết code vẽ biểu đồ bị lỗi:** {e}\n\n*Code gốc của AI (Dành cho Debug):*\n```python\n{clean_code}\n```")

    st.markdown("---")
    st.markdown("💡 **Hệ thống AI đã bóc tách thành công Insight từ CSDL. Xem chi tiết tại các tab bên dưới.**")
    
    if audit_logs:
        with st.expander("🔍 Biên bản Kiểm định Dữ liệu (Auto-Audit Workflow)", expanded=False):
            for log in audit_logs:
                if log["status"] == "error":
                    st.error(log["msg"])
                elif log["status"] == "warning":
                    st.warning(log["msg"])
                else:
                    st.success(log["msg"])
    else:
        st.success("✔️ **Dữ liệu đã được trích xuất an toàn từ CSDL Doanh Nghiệp**")

    tab1, tab2, tab3 = st.tabs(["📊 Báo cáo Phân tích (Insight)", "💡 Đề xuất Chiến lược", "⚙️ Tiến trình SQL"])
    
    with tab1:
        st.markdown(phan_tich if phan_tich else "Không tìm thấy nội dung phân tích.")
            
    with tab2:
        st.markdown(chien_luoc)
        
    with tab3:
        if sql_blocks:
            st.markdown("**Câu lệnh SQL đã được Agent thực thi:**")
            for sql in sql_blocks:
                st.code(sql, language="sql")
                st.markdown("**🗄️ Bảng kết quả truy xuất (Data Preview):**")
                if "SELECT" in sql.upper():
                    try:
                        engine = create_engine(get_db_uri())
                        df_preview = pd.read_sql(sql, engine)
                        st.dataframe(df_preview, use_container_width=True)
                    except Exception as e:
                        st.error(f"⚠️ Lỗi truy xuất CSDL: {e}")
                else:
                    st.info("💡 Câu lệnh này không phải là lệnh truy vấn bảng (SELECT) nên không có Data Preview.")
        else:
            st.info("Agent đã sử dụng dữ liệu ngữ cảnh hoặc tiến trình bị ngắt.")

# Hiển thị tin nhắn CỦA PHIÊN CHAT HIỆN TẠI
current_messages = st.session_state.all_chats.get(st.session_state.current_session_id, [])

for msg in current_messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            render_assistant_response(msg["content"])

if prompt := st.chat_input("VD: Cho tôi insights về địa lý..."):
    st.session_state.all_chats[st.session_state.current_session_id].append({"role": "user", "content": prompt})
    save_history(st.session_state.all_chats)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    agent, status = get_agent()
    
    with st.chat_message("assistant"):
        if agent is None:
            st.error(status)
        else:
            with st.spinner("Đang chạy luồng kiểm định chất lượng dữ liệu..."):
                current_audit = run_data_audit(get_db_uri())
                
            with st.spinner("Agent đang xử lý phân tích và tổng hợp Insight..."):
                try:
                    response = agent.invoke({"input": prompt})
                    answer = response["output"]
                    
                    render_assistant_response(answer, current_audit)
                    
                    st.session_state.all_chats[st.session_state.current_session_id].append({"role": "assistant", "content": answer})
                    save_history(st.session_state.all_chats)
                    
                except Exception as e:
                    error_str = str(e)
                    if "[PHÂN TÍCH]" in error_str or "[BIỂU ĐỒ]" in error_str:
                        extracted_answer = error_str.split("Could not parse LLM output:")[-1].strip()
                        render_assistant_response(extracted_answer, current_audit)
                        
                        st.session_state.all_chats[st.session_state.current_session_id].append({"role": "assistant", "content": extracted_answer})
                        save_history(st.session_state.all_chats)
                    else:
                        st.error(f"Đã có lỗi hệ thống xảy ra: {e}")