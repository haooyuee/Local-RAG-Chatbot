import pandas as pd
import os
import sys
import datetime
from src import chatbot_base, chatbot
from datasets import Dataset
# Function to evaluate as Llama index does not support async evaluation for HFInference API
def generate_responses(test_questions, test_answers):
  outputs = [chat_bot.generate_response(q, pdf_file) for q in test_questions]

  answers = []
  contexts = []
  for output in outputs:
    answers.append(output['answer'])
    contexts.append([doc.page_content for doc in output['source_documents']])
  dataset_dict = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
  }
  if test_answers is not None:
    dataset_dict["ground_truth"] = test_answers
  ds = Dataset.from_dict(dataset_dict)
  return ds

if __name__ == "__main__":
    directory = "data"
    path        =  "D:/GithubLocal/RAG-with-Llama2/config.yaml"
    pdf_file    = "D:/GithubLocal/RAG-with-Llama2/documents/2312.10997.pdf"
    chat_bot    = chatbot.PDFChatBot(config_path=path)

    eval_dataset = pd.read_csv('eval/testset0330.csv')
    test_questions = eval_dataset['question'].values.tolist()
    test_answers = eval_dataset['ground_truth'].values.tolist()

    dataset = generate_responses(test_questions, test_answers)

    now = datetime.datetime.now()
    os.makedirs('data', exist_ok=True)
    file_name = 'ds_' + now.strftime("%Y-%m-%d_%H-%M-%S")
    full_path = os.path.join(directory, file_name)

    dataset.save_to_disk(full_path)
    print(f"DS file saved: {full_path}")
    print(dataset.to_pandas())
    print("Done")
    

