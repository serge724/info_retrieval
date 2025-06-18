import os
import torch
import numpy as np
import pandas as pd
from doc2data.pdf import PDFCollection
from transformers import LayoutXLMProcessor
from transformers import LayoutLMv2ForTokenClassification
from processors import TokenClsProcessorLayoutXLM
from tqdm import tqdm
from utils import train_model, map_predictions

# create folder for results
results_path = 'results/information_extraction/layoutxlm'
os.makedirs(results_path)

# load data
pdf_collection = PDFCollection.load('data/pdf_collection.pickle')
instances_df = pd.read_pickle('processed_data/information_extraction/instances_page_parts.pickle')

# load and process target pages
target_pages = pd.read_csv('data/labels/page_identification_labels.csv')
target_pages = target_pages[~target_pages['no_subs']].reset_index(drop = True)
target_pages = target_pages[~target_pages['foreign_issuer']].reset_index(drop = True)
target_pages.page_nr = target_pages.page_nr - 1
target_pages['instance_id'] = target_pages.file_name + ' - ' + target_pages.page_nr.astype('int').astype('str')
target_pages = target_pages[target_pages.page_label != 'not_relevant']

# ensure that all pages from page identification are also in information extraction dataset
assert target_pages.instance_id.isin(instances_df.page_instance_id).all()

# select only the primary range for infromation extraction
primary_range = target_pages[target_pages.page_label == 'primary_range'].reset_index(drop = True)
instances_df = instances_df[instances_df.page_instance_id.isin(primary_range.instance_id)].reset_index(drop = True)

# prepare data processor
layoutxlm_processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base", apply_ocr=False)
layoutxlm_processor.tokenizer.model_max_length = 512
def processing_function(features, labels, inference_mode):

    image, tokens, bboxes = features
    # truncation disabled since secqueces are split into correct chunks before
    encoding = layoutxlm_processor(images = image, text = tokens, boxes = bboxes, word_labels = labels, padding="max_length", truncation=False)

    encoding['input_ids'] = np.array(encoding['input_ids'])
    encoding['attention_mask'] = np.array(encoding['attention_mask'])
    encoding['bbox'] = np.array(encoding['bbox'])
    encoding['image'] = np.array(encoding['image'][0])
    encoding['labels'] = np.array(encoding['labels'])

    assert len(encoding['input_ids']) <= layoutxlm_processor.tokenizer.model_max_length, 'Check chunk length'

    return encoding

processor = TokenClsProcessorLayoutXLM(
    instances_df,
    processing_function,
    pdf_collection
)

# initialize data loaders
train_data = processor.initialize_split('train_set', batch_size = 4, shuffle = True)
val_data = processor.initialize_split('val_set', batch_size = 4, shuffle = False)
test_data = processor.initialize_split('test_set', batch_size = 4, shuffle = False)

print('Number of batches in training data:', len(train_data))
print('Number of batches in validation data:', len(val_data))
print('Number of batches in testing data:', len(test_data))

# load pretrained layoutxlm
model = LayoutLMv2ForTokenClassification.from_pretrained(
    'microsoft/layoutxlm-base',
    num_labels=len(processor.id2label)
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# train model
training_results = train_model(
    model,
    n_epochs = 5,
    task = 'information_extraction',
    train_data = train_data,
    val_data = val_data,
    learning_rate = 1e-5,
    device = device
)

# visualize training history
training_history = pd.concat([
    pd.DataFrame.from_dict(training_results['train_history'], orient='index').assign(data_split = 'train_set'),
    pd.DataFrame.from_dict(training_results['val_history'], orient='index').assign(data_split = 'val_set'),
])
training_history.to_csv(os.path.join(results_path, 'training_history.csv'), index = True)
training_history[['average_loss', 'data_split']].pivot(columns = 'data_split', values = 'average_loss').plot()
training_history[['accuracy', 'data_split']].pivot(columns = 'data_split', values = 'accuracy').plot()

# load model from best epoch
model = LayoutLMv2ForTokenClassification.from_pretrained(f'models/information_extraction')
model.to(device)

# calculate predictions on test dataset
predictions = []
probabilities = []
labels = []
input_ids = []
with torch.no_grad():
    for batch in tqdm(test_data):
        batch.to(device)
        outputs = model(**batch)
        batch_preds = outputs.logits.argmax(-1)
        batch_probs = outputs.logits.softmax(-1)
        predictions.append(batch_preds.to('cpu').numpy())
        probabilities.append(batch_probs.to('cpu').numpy())
        labels.append(batch['labels'].to('cpu').numpy())
        input_ids.append(batch['input_ids'].to('cpu').numpy())

# postprocess
test_set_df = instances_df[instances_df.data_split == 'test_set'].copy()
test_set_df = test_set_df.reset_index(drop = True)

predictions = [i for j in predictions for i in j]
probabilities = [i for j in probabilities for i in j]
labels = [i for j in labels for i in j]
input_ids = [i for j in input_ids for i in j]

predictions_df = []
for i, row in tqdm(test_set_df.iterrows()):
    pdf_name, page_nr, part = row.instance_id.split(' - ')
    instance = test_set_df[test_set_df.instance_id == row.instance_id].iloc[0]
    page = pdf_collection.pdfs[pdf_name][int(page_nr)]


    mapping_df = map_predictions(
        instance = instance,
        instance_labels = labels[i],
        instance_predictions = predictions[i],
        instance_input_ids = input_ids[i],
        layoutxlm_processor = layoutxlm_processor,
        processor = processor
    )

    mapping_df['id'] = row.instance_id
    mapping_df['file_name'] = pdf_name
    mapping_df['page_nr'] = page_nr
    mapping_df['part'] = part

    predictions_df.append(mapping_df)

# save results
predictions_df = pd.concat(predictions_df, axis = 0)
predictions_df.to_csv(os.path.join(results_path, 'test_set_predictions.csv'), index = False)
