import os
import json
import pickle
import pandas as pd
from tqdm import tqdm
from doc2data.pdf import PDFCollection
from utils import run_llm_pipeline

pdf_collection = PDFCollection.load('data/pdf_collection.pickle')
instances_df = pd.read_pickle('processed_data/information_extraction/instances_full_pages.pickle')

# load and process target pages
target_pages = pd.read_csv('data/labels/page_identification_labels.csv')
target_pages = target_pages[~target_pages['no_subs']].reset_index(drop = True)
target_pages = target_pages[~target_pages['foreign_issuer']].reset_index(drop = True)
target_pages.page_nr = target_pages.page_nr - 1
target_pages['instance_id'] = target_pages.file_name + ' - ' + target_pages.page_nr.astype('int').astype('str')
target_pages = target_pages[target_pages.page_label != 'not_relevant']

# ensure that all pages from page identification are also in information extraction dataset
assert target_pages.instance_id.isin(instances_df.instance_id).all()

# select only the primary range for infromation extraction
primary_range = target_pages[target_pages.page_label == 'primary_range'].reset_index(drop = True)
instances_df = instances_df[instances_df.instance_id.isin(primary_range.instance_id)].reset_index(drop = True)

with open('prompts/information_extraction.json', 'r') as file:
    prompts = json.load(file)

# run experiments on validation set
for prompt_id, prompt_templates in prompts.items():

    if os.path.exists(f'results/information_extraction/chatgpt/{prompt_id}'):
        continue

    print(prompt_id)

    # create folder for results
    os.makedirs(f'results/information_extraction/chatgpt/{prompt_id}')

    # prepare prompt
    prompt = {
        'system_message': prompt_templates['system_prompt'],
        'user_message': prompt_templates['information_extraction']
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
    with open(f'results/information_extraction/chatgpt/{prompt_id}/llm_predictions.pickle', 'wb') as file:
        pickle.dump(llm_result, file)

# run predictions on test set
# create folder for results
os.makedirs(f'results/information_extraction/chatgpt/test_set')

# prepare prompt
prompt = {
    'system_message': prompts['prompt_2']['system_prompt'],
    'user_message': prompts['prompt_2']['information_extraction']
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
with open(f'results/information_extraction/chatgpt/test_set/llm_predictions.pickle', 'wb') as file:
    pickle.dump(llm_result, file)
