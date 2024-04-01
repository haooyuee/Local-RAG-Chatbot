#streamlit run app_demo.py --logger.level=DEBUG
#streamlit run app_demo.py
import streamlit as st
from streamlit_chat import message
from chatbot_base import PDFChatBot
import time
import atexit
import tempfile
import shutil
import os

# 清理函数，用于关闭临时文件并删除
def cleanup():
    if 'tmp_file_path' in st.session_state and os.path.exists(st.session_state['tmp_file_path']):
        os.remove(st.session_state['tmp_file_path'])
# 在程序退出时自动调用清理函数
atexit.register(cleanup)

def show_chat():
    if st.session_state['pdf_chatbot']:
        for i in range(len(st.session_state['bot_history'])-1, -1, -1):
            message(st.session_state["bot_history"][i], key=str(i), seed=3)
            message(st.session_state['user_history'][i], 
                    is_user=True, 
                    key=str(i)+'_user', seed=2)

# 设置主题和标题
st.set_page_config(page_title="PDF Chatbot Q&A", layout="wide")
st.title('RAG Chatbot Q&A')
st.sidebar.header("Intro")
st.sidebar.info(
    '''This is a web application. 
    Enter a question in the text box and press Enter to query and receive answers from ChatBot.'''
)


# 初始化PDFChatBot和聊天历史
if 'pdf_chatbot' not in st.session_state:
    st.session_state.pdf_chatbot = PDFChatBot()

if 'bot_history' not in st.session_state:
    st.session_state.bot_history = []
if 'user_history' not in st.session_state:
    st.session_state.user_history = []

left_column, right_column = st.columns([3, 2])


### ********************* RIGHT SIDE ********************* 
with right_column:
    uploaded_file = st.file_uploader("📁 Upload PDF", type="pdf", key="pdf_uploader")
# 处理PDF上传并显示预览
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        shutil.copyfileobj(uploaded_file, tmp_file)
        st.session_state['tmp_file_path'] = tmp_file.name
    try:
        image = st.session_state.pdf_chatbot.render_file(st.session_state['tmp_file_path'])
        right_column.image(image, caption='PDF Preview', use_column_width=True)
    except Exception as e:
        st.error(f"Error processing PDF file: {str(e)}")
### ********************* RIGHT SIDE ********************* END

### ********************* LEFT SIDE **********************
with left_column:
    with st.form(key='question_form',clear_on_submit=True):
        question = st.text_input("Ask a question about the PDF:", key="question")
        submit_button = st.form_submit_button(label='Send')
    
    chat_container = st.container()
### ********************* LEFT SIDE ********************** END         
if submit_button and question:
    if 'tmp_file_path' in st.session_state:
        st.session_state.user_history.append(question)

        try:
            with st.spinner(text='Generating response...'):
                result = st.session_state.pdf_chatbot.generate_response(
                    question, 
                    st.session_state['tmp_file_path'])
                st.session_state.bot_history.append(result["answer"])
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    else:
        st.error("Please upload a PDF file before asking questions.")

with chat_container:
    show_chat()

