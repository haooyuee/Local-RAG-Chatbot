## Evaluation

### Datasets
HotpotQA and SQuAD (The Stanford Question Answering Dataset)

### Procedure
Please run the notebooks in the following order to generate synthetic PDFs and evaluate on them:

1. **generate_pdf_hotpotqa.ipynb**
   - This notebook generates PDF files related to HotpotQA subset.

2. **generate_pdf_squad.ipynb**
   - This notebook generates PDF files related to SQuAD subset.

3. **eval_hotpotqa.ipynb**
   - This notebook generates outputs of eval model on HotpotQA subset.

4. **eval_squad.ipynb**
   - This notebook generates outputs of eval model on SQuAD subset.

5. **base_hotpotqa.ipynb**
   - This notebook generates outputs of base model on HotpotQA subset.

6. **base_squad.ipynb**
   - This notebook generates outputs of base model on SQuAD subset.

7. **get_metrics.ipynb**
   - This notebook computes metrics, including RAGAS metrics, which requires an OpenAI API key.

8. **combine_csv.ipynb**
   - This notebook combines the splited CSV files containing the metrics.

### Visualization

To visualiza the metrics, please navigate to ./visualization

### Requirements
- Ensure that you have the necessary datasets and dependencies installed for each notebook. Download HotpotQA via https://github.com/allenai/natural-instructions/blob/master/tasks/task170_hotpotqa_answer_generation.json
- Obtain an OpenAI API key to compute RAGAS metrics in `get_metrics.ipynb`.
- Ensure that packages in requirements.txt have been installed.
- Ensure that you have changed the path to your corresponding local path
