import yaml
import fitz
import torch
import gradio as gr
from PIL import Image
from langchain import hub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
#from langchain_core.runnables import RunnablePassthrough 
#from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_core.runnables import RunnableParallel
#TEST
from FakeLLM import FakePromptCopyLLM

class PDFChatBot:
    def __init__(self, config_path="../config.yaml"):
        """
        Initialize the PDFChatBot instance.

        Parameters:
            config_path (str): Path to the configuration file (default is "../config.yaml").
        """
        self.processed = False
        self.page = 0
        self.chat_history = []
        self.config = self.load_config(config_path)
        # Initialize other attributes to None
        self.prompt = None
        self.documents = None
        self.embeddings = None
        self.vectordb = None
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.chain = None

    def load_config(self, file_path):
        """
        Load configuration from a YAML file.
        Parameters: file_path (str): Path to the YAML configuration file.
        Returns:    dict: Configuration as a dictionary.
        """
        with open(file_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
                return config
            except yaml.YAMLError as exc:
                print(f"Error loading configuration: {exc}")
                return None

    def create_prompt_template(self):
        """
        Create a prompt template for the chatbot. (not use for now)
        """
        template = (
            f"As an intelligent assistant with access to a wide range of information, "
                "your goal is to provide detailed and accurate answers by considering "
                "both the context provided from relevant documents and the user's current "
                "question. Use the information from the documents to enrich your response, "
                "ensuring it is both informative and directly addresses the user's inquiry. "
                "\n\n"
                "Relevant documents provide the following context:\n"
                "{{context}}\n\n"
                "User's question:\n"
                "{{question}}\n\n"
                "Given this background, please formulate a comprehensive and detailed answer "
                "that integrates insights from the provided context and directly responds to "
                "the user's query."
            )
        self.prompt = PromptTemplate.from_template(template)
    
    def load_documents(self, file):
        pdf_loader = PyPDFLoader(file.name)
        documents = pdf_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        self.documents = text_splitter.split_documents(documents)

    def load_embeddings(self):
        """
        Load embeddings from LOCAL.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.get("modelEmbeddings"))

    def load_vectordb(self):
        """
        Load the vector database from the documents and embeddings.
        """
        self.vectordb = Chroma.from_documents(self.documents, self.embeddings)

    def load_tokenizer(self):
        """
        Load the tokenizer from LOCAL.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("autoTokenizer"))

    def load_model(self, use_fake_llm = False):
        """
        Load the causal language model from LOCAL.
        """
        self.model = AutoModelForCausalLM.from_pretrained(self.config.get("autoModelForCausalLM"),
            device_map='auto',
            torch_dtype=torch.float16,
            token=True,
            load_in_8bit=False
        )
        if use_fake_llm:
            # set FAKE LLM for Debugging
            self.pipeline = FakePromptCopyLLM()
        else:
            pipe = pipeline(
                model=self.model,
                task='text-generation',
                tokenizer=self.tokenizer,
                max_new_tokens=200
            )
            self.pipeline = HuggingFacePipeline(pipeline=pipe)

    # def format_docs(self, docs):
    #     # for basic chain
    #     return "\n\n".join(doc.page_content for doc in docs)

    def get_chat_history(self, inputs) -> str:
        res = []
        for human, ai in inputs:
            res.append(f"Human:{human}\nAI:{ai}")
        return "\n".join(res)
    
    def create_chain(self):
        """
        Create a Conversational Retrieval Chain
        """
        retriever=self.vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        self.chain = ConversationalRetrievalChain.from_llm(
            self.pipeline,
            retriever=retriever,
            condense_question_llm  = self.pipeline,
            return_source_documents=True,
            verbose=True,
            get_chat_history=self.get_chat_history
        )

        #retriever_with_print = (lambda x: self.print_and_return(retriever(x), label="Retriever Output"))
                # self.rag_chain = (
                #     RunnablePassthrough.assign(context=(lambda x: self.format_docs(x["context"])))
                #     | self.prompt
                #     | self.pipeline
                #     | StrOutputParser()
                # )
                # self.rag_chain_with_source = RunnableParallel(
                #     {"context": retriever, "question": RunnablePassthrough()}
                #     ).assign(answer=self.rag_chain)
    def process_file(self, file):
        """
        Process the uploaded PDF file and initialize necessary components: Tokenizer, VectorDB and LLM.

        Parameters:
            file (FileStorage): The uploaded PDF file.
        """
        #self.create_prompt_template()
        self.load_documents(file)
        self.load_embeddings()
        self.load_vectordb()
        self.load_tokenizer()
        self.load_model()
        self.create_chain()

    def generate_response(self, history, query, file):
        """
        Generate a response based on user query and chat history.

        Parameters:
            history (list): List of chat history tuples.
            query (str): User's query.
            file (FileStorage): The uploaded PDF file.

        Returns:
            tuple: Updated chat history and a space.
        """
        if not query:
            raise gr.Error(message='Submit a question')
        if not file:
            raise gr.Error(message='Upload a PDF')
        if not self.processed:
            self.process_file(file)
            self.processed = True
        
        # 打印提示模板
        #print("Prompt Template: \n", self.prompt)
        #result = self.rag_chain_with_source.invoke(query)
        #print("result: \n", result)
        result = self.chain({"question": query, 'chat_history': self.chat_history}, return_only_outputs=True)
        self.chat_history.append((query, result['answer']))
        self.page = list(result['source_documents'][0])[1][1]['page']

        for char in result['answer']:
            history[-1][-1] += char
        return history, " "

    def render_file(self, file):
        """
        Renders a specific page of a PDF file as an image.

        Parameters:
            file (FileStorage): The PDF file.

        Returns:
            PIL.Image.Image: The rendered page as an image.
        """
        doc = fitz.open(file.name)
        page = doc[self.page]
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image
    
    def add_text(self, history, text):
        """
        Add user-entered text to the chat history.

        Parameters:
            history (list): List of chat history tuples.
            text (str): User-entered text.

        Returns:
            list: Updated chat history.
        """
        if not text:
            raise gr.Error('Enter text')
        history.append((text, ''))
        return history

if __name__ == "__main__":
    chat_bot = PDFChatBot(config_path="../config.yaml")

    # 假设有一个PDF文件已经准备好，这里用'example.pdf'代替
    # 在实际情况中，你需要确保这个文件存在
    pdf_file = gr.File()  # 这里仅为示例，实际应用中需要使用正确的文件对象
    pdf_file.name = "../barlowtwins-CXR.pdf"  # 指定文件名，假设文件已经加载

    # 模拟五轮对话
    queries = [
        "What is the main topic of the document?",
        "Can you explain the key findings?",
        "Are there any notable figures or tables?",
        "How do the authors conclude their research?",
        "Is there any discussion on future work?"
    ]

    # 执行对话
    for i, query in enumerate(queries, start=1):
        print(f"Round {i}: Query = {query}")
        # 更新聊天历史并获取回答（此处假设chat_history已初始化为空列表）
        chat_bot.chat_history= chat_bot.add_text(chat_bot.chat_history, query)
        chat_bot.chat_history, _ = chat_bot.generate_response(chat_bot.chat_history, query, pdf_file)
        print(f"Answer: {chat_bot.chat_history[-1][1]}\n")