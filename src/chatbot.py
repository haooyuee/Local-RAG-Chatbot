import yaml
import fitz
import torch
import streamlit as st
from PIL import Image
import re  # For regular expressions
from langchain import hub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PDFMinerLoader
from langchain.prompts import PromptTemplate
#from langchain_core.runnables import RunnablePassthrough 
#from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank #cohere 4.57
#from langchain_cohere import CohereRerank #cohere v5 have bug
from apikeys import Cohere_API
#from langchain_core.runnables import RunnableParallel
#TEST
from src.FakeLLM import FakePromptCopyLLM
import io

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
    def clean_text(text):
        #text = text.lower()  # Convert to lowercase
        #text = re.sub(r'[^a-z0-9\s-]', '', text)  # Remove non-alphanumeric characters (except space and dash)
        text = re.sub(r'-\n', '', text)  # Remove hyphens at line breaks
        return text
    
    def load_documents(self, file):
        pdf_loader = PDFMinerLoader(file, concatenate_pages = False)
        documents = pdf_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, 
            chunk_overlap=64, 
            add_start_index=True,
            length_function = len,
            separators=["\n\n", "\n", "(?<=[\.?])", "(?<=[\,;])", " "])
        self.documents = text_splitter.split_documents(documents)

    def load_embeddings(self):
        """
        Load embeddings from LOCAL.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

    def creat_retriever(self):
        """
        Load the vector database from the documents and embeddings.
        """
        # self.vectordb = Chroma.from_documents(self.documents, self.embeddings)
        # self.retriever=self.vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        FAISSdb = FAISS.from_documents(self.documents, self.embeddings) 
        FAISS_retriever=FAISSdb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = 5
        # initialize the ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, FAISS_retriever], weights=[0.5, 0.5])
        compressor = CohereRerank(cohere_api_key=Cohere_API, top_n=3)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever)
            

    # def load_tokenizer(self):
    #     """
    #     Load the tokenizer from LOCAL.
    #     """
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("autoTokenizer"))

    def load_model(self, use_fake_llm = False):
        """
        Load the causal language model from LOCAL.
        """
        if use_fake_llm:
            # set FAKE LLM for Debugging
            self.pipeline = FakePromptCopyLLM()
            print("Currently using FAKE LLM.")
            return
        
        self.model = AutoModelForCausalLM.from_pretrained(self.config.get("autoModelForCausalLM"),
            device_map='auto',
            torch_dtype=torch.float16,
            token=True,
            load_in_8bit=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("autoTokenizer"))
        pipe = pipeline(
            model=self.model,
            task='text-generation',
            tokenizer=self.tokenizer,
            max_new_tokens=256
        )
        self.pipeline = HuggingFacePipeline(pipeline=pipe)

    def get_chat_history(self, inputs, max_history_length = 5):
        res = []
        # Make sure the number of chat turns input to the chain is less than max_history_length
        for human, ai in inputs[-max_history_length:]:
            res.append(f"Human:{human}\nAI:{ai}")
        return "\n".join(res)
    
    def create_chain(self):
        """
        Create a Conversational Retrieval Chain
        """
        self.chain = ConversationalRetrievalChain.from_llm(
            self.pipeline,
            retriever=self.retriever,
            condense_question_llm  = self.pipeline,
            return_source_documents=True,
            verbose=True,
            get_chat_history=self.get_chat_history
        )

    def process_file(self, file):
        """
        Process the uploaded PDF file and initialize necessary components: Tokenizer, VectorDB and LLM.
        Parameters:
            file (FileStorage): The uploaded PDF file.
        """
        #self.create_prompt_template()
        self.load_documents(file)
        self.load_embeddings()
        self.creat_retriever()
        #self.load_tokenizer()
        self.load_model()
        self.create_chain()

    def generate_response(self, query, file):
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
            raise st.Error(message='Submit a question')
        if not file:
            raise st.Error(message='Upload a PDF')
        if not self.processed:
            self.process_file(file)
            self.processed = True
        
        result = self.chain({"question": query, 'chat_history': self.chat_history}, return_only_outputs=True)
        self.chat_history.append((query, result["answer"])) 
        self.page = list(result['source_documents'][0])[1][1]['page']
        return result

    def render_file(self, file):
        """
        Renders a specific page of a PDF file as an image.
        Parameters:
            file (FileStorage): The PDF file.
        Returns:
            PIL.Image.Image: The rendered page as an image.
        """
        doc = fitz.open(file)
        page = doc[self.page]
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image

if __name__ == "__main__":
    path        =  "D:\GithubLocal\RAG-with-Llama2\config.yaml"
    
    pdf_file    = "../documents/barlowtwins-CXR.pdf"
    pdf_file    = "D:/GithubLocal/RAG-with-Llama2/documents/barlowtwins-CXR.pdf"
    chat_bot    = PDFChatBot(config_path=path)
    queries     = [
        "What is the main topic of the document?",
        "Can you explain the key findings?",
        "Are there any notable figures or tables?",
        "How do the authors conclude their research?",
        "What is the last questions?"
    ]

    # begin
    for i in range(len(queries)):
        print(f"Round {i}: Query = {queries[i]}")
        answer= chat_bot.generate_response(queries[i], pdf_file)
        print(f"Answer: {answer['answer']}\n")
    print("chathistory ################")
    print(chat_bot.chat_history)
    print("chathistory finish##########")