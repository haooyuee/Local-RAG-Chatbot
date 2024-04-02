import os
import sys
import datetime
from langchain_community.document_loaders import PDFMinerLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from apikeys import OPENAI_API
import os
if __name__ == "__main__":

    os.environ["OPENAI_API_KEY"] = OPENAI_API
    directory = "data"
    loader = PDFMinerLoader("documents/2312.10997.pdf", concatenate_pages = False)
    documents = loader.load()
    #print(documents)

    # generator with openai models
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    critic_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    embeddings = OpenAIEmbeddings()

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    # generate testset
    testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.4, reasoning: 0.2, multi_context: 0.4})
    
    now = datetime.datetime.now()
    os.makedirs('data', exist_ok=True)
    file_name = 'gt_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
    full_path = os.path.join(directory, file_name)

    testset.to_csv('folder_name.csv', index=False)
    print(f"CSV file saved: {full_path}")
    

