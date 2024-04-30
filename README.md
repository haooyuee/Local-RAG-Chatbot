# Local Assistant Chatbot with Memory Through Retrieval-Augmented Generation

This project explores leveraging Retrieval Augmented Generation (RAG) to empower local assistant chatbots for efficient and privacy-aware PDF management. 

Compute resource requirements: 16GB RAM, 6GB GPU memory. 

(Development environment: Lenovo Thinkbook16p Nvidia 3060 6GB)

##
* Langchain Framework
* LLM Model : Llama2, Gemma
* Vector retrieval: ChromaDB, FAISS
* Hugging Face Transformers
* Streamlit
* Evaluation: RAGAS, ROUGE, etc.

## Features 
* Process PDF files and extract information for answering questions.
* Maintain chat history and provide detailed explanations.
* Generate responses using a Conversational Retrieval Chain.
* Display specific pages of PDF files according to the answer.

## Prerequisites
Before running the ChatBot, ensure that you have the required dependencies installed. You can install them using the following command (Ex: Miniconda):
### STEP 1:
```
$conda create -n test python=3.10
```
### STEP 2:
If the Cuda version is 12.1:
```
$pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Else: Install the corresponding torch version
### STEP 3:
```
$pip install -e .
```
### STEP 4:
Due to version issues, Cohere need to be installed separately (2024/04)
```
$pip install cohere==4.57
```

Anticipated possible Langchain bugs and resolutions:
```
# problem Langchain: ModuleNotFoundError: No module named 'pwd'
# solve: https://github.com/langchain-ai/langchain/issues/17514
```

## Configuration 
The ChatBot uses a configuration file (config.yaml) to specify Hugging Face model and embeddings details. Make sure to update the configuration file with the appropriate values if you wanted to try another model or embeddings.

## Usage
1. Upload a PDF file using the "📁 Upload PDF" button.
2. Enter your questions in the text box.
3. Click the "Send" button to submit your question.
4. View the chat history and responses in the interface.

## Running Locally
To run the PDF Interaction ChatBot, execute the following command:

```
streamlit run app_demo.py
```

## Enjoy !😉

