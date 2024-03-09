#streamlit run app_demo.py --logger.level=DEBUG
import streamlit as st
import atexit
import tempfile
import shutil
import os
from chatbot import PDFChatBot

def cleanup():
    # 清理函数，用于关闭临时文件并删除
    if tmp_file is not None:
        tmp_file.close()
        os.remove(tmp_file.name)
# 在程序退出时自动调用清理函数
atexit.register(cleanup)

st.title('PDF Chatbot')
# 初始化PDFChatBot
if 'pdf_chatbot' not in st.session_state:
    # 初始化PDFChatBot
    # 如果不存在，创建一个新的 PDFChatBot 实例并存储在 Session State 中
    st.session_state.pdf_chatbot = PDFChatBot()

# 初始化聊天历史
if 'chat_history' not in st.session_state:
    # 用于存储聊天历史的Session State，如果不存在则初始化为空列表
    st.session_state.chat_history = []

# 文件上传器
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
tmp_file = None  # 初始化临时文件对象

if uploaded_file is not None:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    shutil.copyfileobj(uploaded_file, tmp_file)
    tmp_file_path = tmp_file.name
    tmp_file.close()
    st.session_state['tmp_file_path'] = tmp_file_path

try:
    # 假设 render_file 方法已调整为从上传的文件对象中渲染 PDF 页面，并返回 Pillow 图像对象
    if tmp_file is not None:
        image = st.session_state.pdf_chatbot.render_file(tmp_file)
        st.image(image, caption='PDF Preview')
except Exception as e:
    st.error(f"Error processing PDF file: {str(e)}")

# 文本输入，用于接收用户问题
question = st.text_input("Ask a question about the PDF:")
submit = st.button('Submit')
if submit and question:
    if uploaded_file is not None:
        # 更新聊天历史
        st.session_state.chat_history.append(f"You: {question}")
        # 生成回应
        try:
            if 'tmp_file_path' in st.session_state:
                # 假设 generate_response 方法已调整为接受问题和上传的文件对象，并返回回应
                response = st.session_state.pdf_chatbot.generate_response(st.session_state.chat_history, question, st.session_state['tmp_file_path'])
                st.session_state.chat_history.append(f"Bot: {response}")
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    else:
        st.error("Please upload a PDF file before asking questions.")

# 显示聊天历史
st.write("Chat History:")
for msg in st.session_state.chat_history:
    st.text(msg)