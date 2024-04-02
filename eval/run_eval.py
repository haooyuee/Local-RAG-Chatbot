import os
import sys
import datetime
from datasets import load_from_disk
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from apikeys import OPENAI_API

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = OPENAI_API
    directory = "data"
    loaded_dataset = load_from_disk("data\ds_2024-03-31_11-51-32") 
    result = evaluate(
        loaded_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )
    res_df = result.to_pandas()

    now = datetime.datetime.now()
    os.makedirs('data', exist_ok=True)
    file_name = 'res_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
    full_path = os.path.join(directory, file_name)

    res_df.to_csv(full_path, index=False)
    print(f"CSV file saved: {full_path}")
    print(res_df)
    
