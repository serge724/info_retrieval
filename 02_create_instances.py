# Author: Sergej Levich
# Journal article: Sergej Levich and Lucas Knust, International Journal of Accounting Information Systems, https://doi.org/10.1016/j.accinf.2025.100750

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from doc2data.pdf import PDFCollection

# setup output directories
output_path_identification = 'processed_data/page_identification'
os.makedirs(output_path_identification)
output_path_extraction = 'processed_data/information_extraction'
os.makedirs(output_path_extraction)

# load pdf collection
pdf_collection = PDFCollection.load('data/pdf_collection.pickle')

# load data split
data_split = pd.read_csv('data/data_split.csv')

# get overview for all pages
page_overview = pdf_collection.overview.copy()

# load page identification labels
target_pages = pd.read_csv('data/labels/page_identification_labels.csv')

# check if all documents are present
assert set(page_overview.file_name) == set(target_pages.file_name)

# check counts
print(
    pd.DataFrame(
        target_pages
        [['foreign_issuer', 'no_subs', 'page_label']]
        .value_counts(dropna = False)
    )
    .sort_values(by='foreign_issuer')
)

# adjust pages numbers to begin with 0
target_pages.page_nr = target_pages.page_nr - 1

# add further instance information
identification_instances = page_overview.merge(
    target_pages[['file_name', 'company', 'foreign_issuer', 'no_subs']].drop_duplicates(),
    how = 'left',
    on = 'file_name'
)

# check join
assert len(identification_instances) == len(page_overview)

# remove reports that do not list subsidiaries (these reports are still used for training as negative samples)
target_pages = target_pages[~target_pages['no_subs']].reset_index(drop = True)

# remove foreign issuers
target_pages = target_pages[~target_pages['foreign_issuer']].reset_index(drop = True)

# ensure that each target page has a valid page number
target_pages.page_nr = target_pages.page_nr.astype(int)
assert not any(target_pages.page_nr.isna()) & all(target_pages.page_nr >= 0)

# add instance ids
identification_instances['instance_id'] = identification_instances.file_name + ' - ' + identification_instances.page_nr.astype('str')
target_pages['instance_id'] = target_pages.file_name + ' - ' + target_pages.page_nr.astype('str')

# join labels
identification_instances = identification_instances.merge(
    target_pages[['instance_id', 'page_label']],
    how = 'left',
    on = 'instance_id'
)

# check join
assert len(identification_instances) == len(page_overview)

# remove foreign issuers from dataset
identification_instances = identification_instances[~identification_instances.foreign_issuer].reset_index(drop = True)

# reformat label column
identification_instances = identification_instances.rename(columns = {'page_label': 'label'})
identification_instances.label = identification_instances.label.fillna("not_relevant")
identification_instances.label = identification_instances.label.map({'primary_range': True, 'secondary_range': True, 'not_relevant': False})

# add tokens and bounding boxes
identification_instances['tokens'] = pd.NA
identification_instances['bboxes'] = pd.NA
for i, row in tqdm(identification_instances.iterrows(), total = len(identification_instances)):
    tokens = pdf_collection.pdfs[row.file_name][(row.page_nr)].read_contents('tokens')
    identification_instances.at[i, 'tokens'] = [token['text'] for token in tokens]
    identification_instances.at[i, 'bboxes'] = [token['bbox'] for token in tokens]

# add data split
identification_instances = identification_instances.merge(
    data_split,
    how = 'left',
    on = 'company'
)

# check counts
print(
    pd.DataFrame(
        identification_instances
        .assign(is_na_page_nr=lambda x: x['page_nr'].isna())
        [['foreign_issuer', 'no_subs', 'label', 'is_na_page_nr']]
        .value_counts(dropna = False)
    )
    .sort_values(by=['foreign_issuer', 'no_subs'])
)

# select relevant columns
identification_instances = identification_instances[[
    'instance_id',
    'file_name',
    'company',
    'page_nr',
    'label',
    'tokens',
    'bboxes',
    'data_split'
]]

# save instances
identification_instances.to_pickle(f"{output_path_identification}/instances.pickle")

# load page information extraction labels
with open('data/labels/bbox_labels.json', 'r') as file:
    bbox_labels = json.load(file)

# load preprocessor for LayoutXLM
from transformers import LayoutXLMProcessor
layoutxlm_processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base", apply_ocr=False)

# generate instances for information extraction
records_full_page = []
records_page_parts = []
for file_name, page in tqdm(bbox_labels.items()):
    for page_nr, tokens in page.items():

        # create instance id
        id = file_name + ' - ' + str(page_nr)

        # tokenize text using layoutxlm processor
        tokenized = layoutxlm_processor.tokenizer(
            text = [i['text'] for i in tokens],
            boxes = [i['bbox'] for i in tokens],
            return_offsets_mapping = True
        )
        tokenized_df = pd.DataFrame({
            'token_id': tokenized['input_ids'],
            'bbox': tokenized['bbox'],
            'word_id': tokenized.word_ids()
        })
        # remove start and end tokens
        tokenized_df = tokenized_df[~tokenized_df.token_id.isin([0, 2])]
        tokenized_df.word_id = tokenized_df.word_id.astype('int')
        # create flag to check if next word is a new word
        tokenized_df['new_word_follows'] = tokenized_df['word_id'] != tokenized_df['word_id'].shift(-1)

        # Generate instances for full pages (for LLMs)
        # map labels to tokens
        token_ids = [i['id'] for i in tokens]
        token_labels = [i['label'] for i in tokens]

        # label_map = {i['id']:i['label'] for i in tokens}
        # token_labels = [label_map.get(id, 'Other') for id in token_ids]

        # check validity of labels
        token_labels = [i if i != 'LE_C' else 'LE' for i in token_labels]
        assert len(token_labels) == len(token_ids) == len(tokens)
        assert set(token_labels).issubset({'C', 'LE', 'Other', 'Own'})

        records_full_page.append({
            'instance_id': id,
            'file_name': file_name,
            'page_nr': int(page_nr),
            'n_subtokens': tokenized_df.shape[0],
            'n_tokens': len(token_ids),
            'subtoken_ids': tokenized_df.word_id.astype('int').tolist(),
            'token_ids': token_ids,
            'tokens': [i['text'] for i in tokens],
            'bboxes': [i['bbox'] for i in tokens],
            'labels': token_labels
        })

        # Generate instances for page parts (for LayoutXLM)
        # calculate necessary number of splits to reduce page length
        n_splits = tokenized_df.shape[0] / (512 - 2) # account for additional start/end tokens
        n_splits = int(np.ceil(n_splits))

        page_parts = []
        prev_split_idx = 0
        next_split_idx = 0
        for i in range(n_splits):
            next_split_idx = prev_split_idx + (512-2)
            if tokenized_df[prev_split_idx:next_split_idx].new_word_follows.iloc[-1]:
                page_parts.append(tokenized_df[prev_split_idx:next_split_idx])
            else:
                if i+1 == n_splits:
                    raise RuntimeError('Check last word of processed part of the page!')
                last_words = np.where(tokenized_df[prev_split_idx:next_split_idx].new_word_follows)[0]
                next_split_idx = prev_split_idx + last_words[-1] + 1
                page_parts.append(tokenized_df[prev_split_idx:next_split_idx])

            prev_split_idx = next_split_idx

        assert pd.concat(page_parts).equals(tokenized_df)
        assert all([i.shape[0] <= (512-2) for i in page_parts])

        for i, part in enumerate(page_parts):

            part_tokens = [i for i in tokens if i['id'] in part.word_id.astype('int').unique()]
            part_token_ids = [i['id'] for i in part_tokens]
            part_token_labels = [i['label'] for i in part_tokens]

            # check validity of labels
            part_token_labels = [i if i != 'LE_C' else 'LE' for i in part_token_labels]
            assert len(part_token_labels) > 0
            assert len(part_token_labels) == len(part_token_ids) == len(part_tokens)
            assert set(part_token_labels).issubset({'C', 'LE', 'Other', 'Own'})

            records_page_parts.append({
                'instance_id': f"{id} - {i}",
                'page_instance_id': id,
                'file_name': file_name,
                'page_nr': int(page_nr),
                'part': i,
                'n_subtokens': part.shape[0],
                'n_tokens': len(part_token_ids),
                'subtoken_ids': part.word_id.astype('int').tolist(),
                'token_ids': part_token_ids,
                'tokens': [i['text'] for i in part_tokens],
                'bboxes': [i['bbox'] for i in part_tokens],
                'labels': part_token_labels
            })

instances_df_page_parts = pd.DataFrame(records_page_parts)
instances_df_full_pages = pd.DataFrame(records_full_page)

# add further instance information
instances_df_page_parts = instances_df_page_parts.merge(
    target_pages[['file_name', 'company']].drop_duplicates(),
    how = 'left',
    on = 'file_name'
)
instances_df_full_pages = instances_df_full_pages.merge(
    target_pages[['file_name', 'company']].drop_duplicates(),
    how = 'left',
    on = 'file_name'
)

# add data split
instances_df_page_parts = instances_df_page_parts.merge(
    data_split,
    how = 'left',
    on = 'company'
)
instances_df_full_pages = instances_df_full_pages.merge(
    data_split,
    how = 'left',
    on = 'company'
)
# check counts
print(instances_df_page_parts.data_split.value_counts(dropna = False))
print(instances_df_full_pages.data_split.value_counts(dropna = False))

# save instances
instances_df_page_parts.to_pickle(f"{output_path_extraction}/instances_page_parts.pickle")
instances_df_full_pages.to_pickle(f"{output_path_extraction}/instances_full_pages.pickle")
