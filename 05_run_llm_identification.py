# Author: Sergej Levich
# Journal article: Sergej Levich and Lucas Knust, International Journal of Accounting Information Systems, https://doi.org/10.1016/j.accinf.2025.100750

import os
import json
import pickle
import pandas as pd
from tqdm import tqdm
from doc2data.pdf import PDFCollection
from utils import run_llm_pipeline

# load data
pdf_collection = PDFCollection.load('data/pdf_collection.pickle')
instances_df = pd.read_pickle('processed_data/page_identification/instances.pickle')

# load prompts
with open('prompts/page_identification.json', 'r') as file:
    prompts = json.load(file)

# run experiments on validation set
for prompt_id, prompt_templates in prompts.items():

    if os.path.exists(f'results/page_identification/chatgpt/{prompt_id}'):
        continue

    print(prompt_id)

    # create folder for results
    os.makedirs(f'results/page_identification/chatgpt/{prompt_id}')

    # prepare prompt
    prompt = {
        'system_message': prompt_templates['system_prompt'],
        'user_message': prompt_templates['page_identification']
    }

    # query llm
    llm_result = run_llm_pipeline(
        prompt = prompt,
        instances_df = instances_df,
        pdf_collection = pdf_collection,
        data_split = 'val_set',
        model = 'gpt-4o-2024-05-13',
        temperature = 0
    )

    # save results
    with open(f'results/page_identification/chatgpt/{prompt_id}/llm_predictions.pickle', 'wb') as file:
        pickle.dump(llm_result, file)

# run predictions on test set
# create folder for results
os.makedirs(f'results/page_identification/chatgpt/test_set')

# prepare prompt
prompt = {
    'system_message': prompts['prompt_4']['system_prompt'],
    'user_message': prompts['prompt_4']['page_identification']
}

# query llm
llm_result = run_llm_pipeline(
    prompt = prompt,
    instances_df = instances_df,
    pdf_collection = pdf_collection,
    data_split = 'test_set',
    model = 'gpt-4o-2024-05-13',
    temperature = 0
)

# save results
with open(f'results/page_identification/chatgpt/test_set/llm_predictions.pickle', 'wb') as file:
    pickle.dump(llm_result, file)
