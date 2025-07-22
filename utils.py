# Author: Sergej Levich
# Journal article: Sergej Levich and Lucas Knust, International Journal of Accounting Information Systems, https://doi.org/10.1016/j.accinf.2025.100750

import time
import torch
import evaluate
import openai
import numpy as np
import pandas as pd
from io import StringIO
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm
from doc2data.utils import denormalize_bbox

def draw_bounding_boxes(image, bounding_boxes, bbox_color = 'red', bbox_width = 1):
    """Draws bounding boxes."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_copy = image.copy()

    # check if only one bounding box was provided
    if isinstance(bounding_boxes[0], float):
        bounding_boxes = [bounding_boxes]

    bounding_boxes = np.array(bounding_boxes)

    for box in bounding_boxes:
        if all(box <= 1.1):
            box = denormalize_bbox(box, image.width, image.height)

        draw = ImageDraw.Draw(image_copy)
        draw.rectangle(box, outline = bbox_color, width = bbox_width)

    return image_copy

def train_model(model, n_epochs, task, train_data, val_data, learning_rate=1e-5, device = 'cpu'):

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # run taining loop
    train_history = {}
    val_history = {}
    best_val_loss = 1e6
    global_step = 0
    for epoch in range(n_epochs):
        if not model.training:
            model.train() # set to training mode

        cls_metrics = evaluate.combine(['accuracy'])
        total_loss = 0.0
        for batch in tqdm(train_data):

            # forward
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item()
            batch_predictions = outputs.logits.argmax(-1)

            # backward
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if task == 'information_extraction':
                flattened_adjusted_predictions = [
                    p.item()
                    for prediction, label in zip(batch_predictions, batch['labels'])
                    for p, l in zip(prediction, label)
                    if l != -100
                ]
                flattened_adjusted_labels = [
                    l.item()
                    for prediction, label in zip(batch_predictions, batch['labels'])
                    for p, l in zip(prediction, label)
                    if l != -100
                ]
                cls_metrics.add_batch(predictions = flattened_adjusted_predictions, references = flattened_adjusted_labels)

            elif task == 'page_identification':
                cls_metrics.add_batch(predictions = batch_predictions, references = batch['labels'])

        print(f"Epoch {epoch} | Global step {global_step}")
        train_metrics = cls_metrics.compute()
        train_history[epoch] = train_metrics
        train_metrics['average_loss'] = total_loss / len(train_data)
        print('Train metrics: ', end='')
        for k, v in train_metrics.items():
            print(f"{k}: {round(v, 5)} | ", end='')
        print()
        model.eval()
        val_metrics = evaluate_model(
            model = model,
            task = task,
            data_loader = val_data,
            device = device,
            return_predictions = False,
            disable_tqdm = False
        )
        val_history[epoch] = val_metrics
        print('Validation metrics: ', end='')
        for k, v in val_metrics.items():
            print(f"{k}: {round(v, 5)} | ", end='')
        print()

        if val_metrics['average_loss'] < best_val_loss:
            print(f'Saving model from epoch {epoch}...')
            model.save_pretrained(f"models/{task}")
            best_val_loss = val_metrics['average_loss']

    return {
        'model': model,
        'train_history': train_history,
        'val_history': val_history
    }

def evaluate_model(
    model,
    task,
    data_loader,
    device,
    return_predictions = False,
    disable_tqdm = False
):
    assert not model.training, "Model not in evaluation mode."

    # initialize metrics
    cls_metrics = evaluate.combine(['accuracy'])

    # set the model to evaluation mode
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", disable = disable_tqdm):

            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item()

            batch_predictions = outputs.logits.argmax(-1)

            if task == 'information_extraction':
                flattened_adjusted_predictions = [
                    p.item()
                    for prediction, label in zip(batch_predictions, batch['labels'])
                    for p, l in zip(prediction, label)
                    if l != -100
                ]
                flattened_adjusted_labels = [
                    l.item()
                    for prediction, label in zip(batch_predictions, batch['labels'])
                    for p, l in zip(prediction, label)
                    if l != -100
                ]
                cls_metrics.add_batch(predictions = flattened_adjusted_predictions, references = flattened_adjusted_labels)

            elif task == 'page_identification':
                cls_metrics.add_batch(predictions = batch_predictions, references = batch['labels'])

    eval_dict = cls_metrics.compute()
    eval_dict['average_loss'] = total_loss / len(data_loader)

    return eval_dict


def map_predictions(instance, instance_labels, instance_predictions, instance_input_ids, layoutxlm_processor, processor):

    len(instance_input_ids)
    unpadded_length = len(instance.subtoken_ids)
    unpadded_length

    # create mapping dataframe on subtoken level
    mapping_df = pd.DataFrame({
        'token_id': instance.subtoken_ids,
        'input_ids': instance_input_ids[1:(unpadded_length+1)], # account for start and end token
        'label_data_loader': instance_labels[1:(unpadded_length+1)],
        'prediction': instance_predictions[1:(unpadded_length+1)]
    })

    # aggregate to token level
    mapping_df = mapping_df.groupby('token_id')[['input_ids', 'label_data_loader', 'prediction']].agg(list).reset_index()
    mapping_df['label'] = instance.labels
    mapping_df['token'] = instance.tokens
    mapping_df['decoded_token'] = mapping_df['input_ids'].apply(layoutxlm_processor.tokenizer.decode)

    # use the prediction of the first subtoken for the entire token
    mapping_df['prediction'] = [processor.id2label[i[0]] for i in mapping_df.prediction]

    return mapping_df

def prompt_llm(system_message, user_message, page_text, model, temperature):

    response = openai.chat.completions.create(
      model=model,
      temperature = temperature,
      messages=[
          {
              "role": "system",
              "content": system_message
          },
          {
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": user_message
                  },
                  {
                      "type": "text",
                      "text": f"Here is the text: \n {page_text}"
                  },
              ]
          }
      ],
      max_tokens=4096,
    )

    return response

def run_llm_pipeline(prompt, instances_df, pdf_collection, data_split, model, temperature):

    instances_df_split = instances_df[instances_df.data_split == data_split]

    responses = []
    usages = []

    for i, instance in tqdm(instances_df_split.iterrows(), total = len(instances_df_split)):

        page = pdf_collection.pdfs[instance.file_name][instance.page_nr]
        page_text = page.read_contents('text')

        success = False
        while not success:
            try:
                response = prompt_llm(
                    system_message = prompt['system_message'],
                    user_message = prompt['user_message'],
                    page_text = page_text,
                    model = model,
                    temperature = 0

                )
                success = True

            except Exception as e:
                print(e)
                time.sleep(1)

        responses.append({
            'file_name': instance.file_name,
            'page_nr': instance.page_nr,
            'response': response
        })
        usages.append(response.usage.model_dump())

    return {
        'data_split': data_split,
        'model': model,
        'temperature': temperature,
        'responses': responses,
        'usages': usages,
        'prompt': prompt
    }

def parse_markdown(markdown_string):

    if markdown_string.startswith('```markdown'):
        # pemove markdown tag and leading/trailing whitespace
        markdown_string = markdown_string.replace('markdown\n', '').replace('`', '').strip()
    else:
        markdown_string = markdown_string.strip()

    # convert markdown table to CSV format
    csv_table = markdown_string.split('\n')[:1] + markdown_string.split('\n')[2:]
    # csv_table = markdown_string.split('\n')[2:]
    csv_table = '\n'.join(csv_table)

    # read CSV format string
    df = pd.read_csv(StringIO(csv_table), sep="|", skipinitialspace=True)

    if df.shape == (0,1):
        return pd.DataFrame([{'Company': csv_table, 'Country': 'NA', 'Percentage Ownership': 'NA'}])

    # drop empty columns generated by the leading and trailing pipes
    df.drop(df.columns[[0, -1]], axis=1, inplace=True)

    # rename columns to remove extra spaces
    df.columns = [col.strip() for col in df.columns]

    return df
