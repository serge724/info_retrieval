import os
import torch
import numpy as np
import pandas as pd
from doc2data.pdf import PDFCollection
from transformers import LayoutXLMProcessor
from transformers import LayoutLMv2ForSequenceClassification
from processors import SequenceClsProcessorLayoutXLM
from tqdm import tqdm
from utils import train_model

# create folder for results
results_path = 'results/page_identification/layoutxlm'
os.makedirs(results_path)

# load data
pdf_collection = PDFCollection.load('data/pdf_collection.pickle')
instances_df = pd.read_pickle('processed_data/page_identification/instances.pickle')

# apply subsampling to training and validation sets
train_val_instances = instances_df[instances_df.data_split.isin(['train_set', 'val_set'])]
instances_df = pd.concat(
    [
        # all positive instances for train / val
        train_val_instances[train_val_instances.label],
        # subsampled negative instances for train / val
        train_val_instances[~train_val_instances.label].sample(n=len(train_val_instances[train_val_instances.label])),
        # full test set
        instances_df[(instances_df.data_split == 'test_set')]
    ],
    axis = 0
)
instances_df = instances_df.reset_index(drop = True)
instances_df.to_pickle('processed_data/page_identification/subsampled.pickle')

# prepare data processor
layoutxlm_processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base", apply_ocr=False)
layoutxlm_processor.tokenizer.model_max_length = 512
def processing_function(features, label, inference_mode):

    image, tokens, bboxes = features
    encoding = layoutxlm_processor(images = image, text = tokens, boxes = bboxes, padding="max_length", truncation=True)

    encoding['input_ids'] = np.array(encoding['input_ids'])
    encoding['attention_mask'] = np.array(encoding['attention_mask'])
    encoding['bbox'] = np.array(encoding['bbox'])
    encoding['image'] = np.array(encoding['image'][0])
    encoding['labels'] = np.array(label)

    return encoding

processor = SequenceClsProcessorLayoutXLM(
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
model = LayoutLMv2ForSequenceClassification.from_pretrained(
    'microsoft/layoutxlm-base',
    num_labels=len(processor.id2label)
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# train model
training_results = train_model(
    model,
    n_epochs = 5,
    task = 'page_identification',
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
model = LayoutLMv2ForSequenceClassification.from_pretrained(f'models/page_identification')
model.to(device)

# calculate predictions on full test dataset
predictions = []
probabilities = []
with torch.no_grad():
    for batch in tqdm(test_data):
        batch.to(device)
        outputs = model(**batch)
        batch_preds = outputs.logits.argmax(-1)
        batch_probs = outputs.logits.softmax(-1)
        predictions.append(batch_preds.to('cpu').numpy())
        probabilities.append(batch_probs.to('cpu').numpy())

# postprocess and save results
test_set_df = instances_df[instances_df.data_split == 'test_set'].copy()
test_set_df = test_set_df.reset_index(drop = True)
labels = test_set_df.label.values

predictions = np.concatenate(predictions, axis = 0)
probabilities = np.concatenate(probabilities, axis = 0)
probabilities = probabilities[:,1]

test_set_df['prediction'] = predictions
test_set_df['probability'] = probabilities
test_set_df.drop(columns = ['tokens', 'bboxes']).to_csv(os.path.join(results_path, 'test_set_predictions.csv'), index = False)
