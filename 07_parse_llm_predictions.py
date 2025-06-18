import pickle
import pandas as pd
from doc2data.pdf import PDFCollection
from utils import parse_markdown

pdf_collection = PDFCollection.load('data/pdf_collection.pickle')

# load and postprocess page identification predictions for validation set
instances_df = pd.read_pickle('processed_data/page_identification/instances.pickle')

prompt_ids = ['prompt_1', 'prompt_2', 'prompt_3', 'prompt_4']
llm_results = []
for prompt_id in prompt_ids:
    with open(f'results/page_identification/chatgpt/{prompt_id}/llm_predictions.pickle', 'rb') as file:
        llm_result = pickle.load(file)

    val_instances_df = instances_df[instances_df.data_split == 'val_set'].copy()
    val_instances_df['response'] = [i['response'].choices[0].message.content for i in llm_result['responses']]
    val_instances_df['prediction'] = [True if i == 'Yes' else False for i in val_instances_df['response']]
    val_instances_df = val_instances_df.reset_index(drop = True)
    val_instances_df = pd.concat([val_instances_df, pd.DataFrame(llm_result['usages'])], axis = 1)
    val_instances_df = val_instances_df.drop(columns = ['tokens', 'bboxes'])
    val_instances_df['prompt_id'] = prompt_id

    llm_results.append(val_instances_df)

llm_results = pd.concat(llm_results, axis = 0).reset_index(drop = True)
llm_results.to_csv('results/page_identification/chatgpt/llm_predictions_val_set.csv', index = False)

# load and postprocess page identification predictions for test set
with open(f'results/page_identification/chatgpt/test_set/llm_predictions.pickle', 'rb') as file:
    llm_result = pickle.load(file)

test_instances_df = instances_df[instances_df.data_split == 'test_set'].copy()
test_instances_df['response'] = [i['response'].choices[0].message.content for i in llm_result['responses']]
test_instances_df['prediction'] = [True if i == 'Yes' else False for i in test_instances_df['response']]
test_instances_df = test_instances_df.reset_index(drop = True)
test_instances_df = pd.concat([test_instances_df, pd.DataFrame(llm_result['usages'])], axis = 1)
test_instances_df = test_instances_df.drop(columns = ['tokens', 'bboxes'])
test_instances_df['prompt_id'] = 'prompt_4'

test_instances_df.to_csv('results/page_identification/chatgpt/llm_predictions_test_set.csv', index = False)

# load and postprocess information extraction predictions for validation set
instances_df = pd.read_pickle('processed_data/information_extraction/instances_full_pages.pickle')

# load and process target pages
target_pages = pd.read_csv('data/labels/page_identification_labels.csv')
target_pages = target_pages[~target_pages['no_subs']].reset_index(drop = True)
target_pages = target_pages[~target_pages['foreign_issuer']].reset_index(drop = True)
target_pages.page_nr = target_pages.page_nr - 1
target_pages['instance_id'] = target_pages.file_name + ' - ' + target_pages.page_nr.astype('int').astype('str')
target_pages = target_pages[target_pages.page_label != 'not_relevant']

# select only the primary range for infromation extraction
primary_range = target_pages[target_pages.page_label == 'primary_range'].reset_index(drop = True)
instances_df = instances_df[instances_df.instance_id.isin(primary_range.instance_id)].reset_index(drop = True)

# postprocess predictions and export results for manual evaluation
prompt_ids = ['prompt_1', 'prompt_2', 'prompt_3']
llm_results = []
for prompt_id in prompt_ids:
    with open(f'results/information_extraction/chatgpt/{prompt_id}/llm_predictions.pickle', 'rb') as file:
        llm_result = pickle.load(file)

    val_instances_df = instances_df[instances_df.data_split == 'val_set'].copy()
    val_instances_df['responses'] = [i['response'].choices[0].message.content for i in llm_result['responses']]

    parsed_responses = []
    for i, row in val_instances_df.iterrows():

        # fix parsing error
        if row.instance_id == 'WISeKey International Holding Ltd_en_2021.pdf - 52':
            adjusted = row.responses
            adjusted = adjusted.replace('c|o', 'c/o')
            parsed_response = parse_markdown(adjusted)
        else:
            parsed_response = parse_markdown(row.responses)
        parsed_response['file_name'] = row.file_name
        parsed_response['page_nr'] = row.page_nr
        parsed_responses.append(parsed_response)

    parsed_responses = pd.concat(parsed_responses, axis = 0).reset_index(drop = True)
    parsed_responses['prompt_id'] = prompt_id

    llm_results.append(parsed_responses)

llm_results = pd.concat(llm_results, axis = 0).reset_index(drop = True)
llm_results.to_csv('results/information_extraction/chatgpt/llm_predictions_val_set.csv', index = False)

# load and postprocess page identification predictions for test set
with open(f'results/information_extraction/chatgpt/test_set/llm_predictions.pickle', 'rb') as file:
    llm_result = pickle.load(file)

test_instances_df = instances_df[instances_df.data_split == 'test_set'].copy()
test_instances_df['response'] = [i['response'].choices[0].message.content for i in llm_result['responses']]
parsed_responses = []
for i, row in test_instances_df.iterrows():

    parsed_response = parse_markdown(row.response)
    parsed_response['file_name'] = row.file_name
    parsed_response['page_nr'] = row.page_nr
    parsed_responses.append(parsed_response)

parsed_responses = pd.concat(parsed_responses, axis = 0).reset_index(drop = True)
parsed_responses['prompt_id'] = 'prompt_2'
parsed_responses.to_csv('results/information_extraction/chatgpt/llm_predictions_test_set.csv', index = False)
