# Local Assistant Chatbot with Memory Through Retrieval-Augmented Generation
Ongoing project ... 


This project explores leveraging Retrieval Augmented Generation (RAG) to empower local assistant chatbots for efficient and privacy-aware PDF management. By integrating RAG with local pre-trained large language models (LLMs) within the Langchain framework, we aim to overcome limitations in task-specific performance and computational efficiency. This local ChatPDF-like chatbot will prioritize both retrieval and generation quality, evaluated through frameworks like RGB and RAGAS alongside human assessment. 

## Technologies Used 🚀
* Langchain
* Model : Llama2, Gemma
* Vector database : ChromaDB, FAISS
* Hugging Face Transformers
* Streamlit

## Features ⭐
* Process PDF files and extract information for answering questions.
* Maintain chat history and provide detailed explanations.
* Generate responses using a Conversational Retrieval Chain.
* Display specific pages of PDF files according to the answer.

## Prerequisites 📋
Before running the ChatBot, ensure that you have the required dependencies installed. You can install them using the following command:
```
pip install -r requirements.txt
```

## Configuration ⚙️
The ChatBot uses a configuration file (config.yaml) to specify Hugging Face model and embeddings details. Make sure to update the configuration file with the appropriate values if you wanted to try another model or embeddings.

## Usage 📚
1. Upload a PDF file using the "📁 Upload PDF" button.
2. Enter your questions in the text box.
3. Click the "Send" button to submit your question.
4. View the chat history and responses in the interface.

## Running Locally 💻
To run the PDF Interaction ChatBot, execute the following command:

```
streamlit run app_demo.py
```

## License
This project is licensed under the [Apache License 2.0](https://github.com/Niez-Gharbi/PDF-RAG-with-Llama2-and-Gradio/blob/main/LICENSE).
