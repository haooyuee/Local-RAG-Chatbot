import streamlit as st
from chatbot import PDFChatBot

# 创建 PDFChatBot 实例
pdf_chatbot = PDFChatBot()

# Streamlit 应用的布局
st.title('PDF Chatbot')

# 上传PDF文件
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
if uploaded_file is not None:
    # 这里可以添加代码以处理PDF文件，例如显示PDF的内容或第一页等

    # 模拟处理文件（这里需要你根据实际情况填充逻辑）
    # 注意：这里假设`pdf_chatbot.render_file`方法已调整为适用于Streamlit
    image = pdf_chatbot.render_file(uploaded_file)
    st.image(image, caption='PDF Preview')

# 文本输入用于聊天
user_input = st.text_input("Ask a question about the PDF:")

# 当用户提交问题时
if st.button('Submit'):
    if uploaded_file is not None and user_input:
        # 这里调用你的处理函数来生成回应
        # 注意：这里假设`pdf_chatbot.generate_response`方法已调整为适用于Streamlit
        response = pdf_chatbot.generate_response(user_input, uploaded_file)
        st.write(response)
    else:
        st.write("Please upload a PDF file and enter a question.")

# 运行 Streamlit 应用
# 在终端中使用命令：streamlit run app_streamlit.py