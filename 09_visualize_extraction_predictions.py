# Author: Sergej Levich
# Journal article: Sergej Levich and Lucas Knust, International Journal of Accounting Information Systems, https://doi.org/10.1016/j.accinf.2025.100750

import os
import pickle
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from doc2data.pdf import PDFCollection
from utils import draw_bounding_boxes

# define colors for bounding boxes
color_dict = {
    "Other": "gray",
    "LE": "red",
    "C": "blue",
    "Own": "green"
}

# concatenate images horizontally
def concat_images(left_image, right_image):

    left_image_width, left_image_height = left_image.size
    right_image_width, right_image_height = right_image.size

    if left_image_height != right_image_height:
        raise ValueError("Images should have the same height")

    concat_image = Image.new("RGB", (left_image_width + right_image_width, left_image_height))
    concat_image.paste(left_image, (0, 0))
    concat_image.paste(right_image, (left_image_width, 0))

    return concat_image

# add text to image
def add_text_to_image(image, text):
    draw = ImageDraw.Draw(image)

    # Specify the font, size, and color
    font = ImageFont.load_default()
    font_size = 40
    font_color = (0, 0, 0)  # White

    # Draw the text
    position = (10, 10)  # Top-left position
    draw.text(position, text, font_color, font=font)

    return image


# load data
pdf_collection = PDFCollection.load('data/pdf_collection.pickle')
extraction_instances = pd.read_pickle('processed_data/information_extraction/instances_page_parts.pickle')
test_set_predictions = pd.read_csv('results/information_extraction/layoutxlm/test_set_predictions.csv')

# setup output directory
extraction_visualization_path = 'results/visualization'
os.makedirs(extraction_visualization_path)

# draw labels and predictions and write to disk
for id in tqdm(test_set_predictions.id.unique()):

    instance = extraction_instances[extraction_instances.instance_id == id].iloc[0]
    predictions = test_set_predictions[test_set_predictions.id == id]
    file_name = instance.file_name
    page_nr = instance.page_nr

    page = pdf_collection.pdfs[file_name][page_nr]

    labels_image = page.read_contents('page_image', dpi = 100, force_rgb = True)
    for j, lab in enumerate(instance.labels):
        labels_image = draw_bounding_boxes(labels_image, [instance.bboxes[j]], bbox_color = color_dict[lab], bbox_width = 2)

    predictions_image = page.read_contents('page_image', dpi = 100, force_rgb = True)
    for j, lab in enumerate(predictions.prediction):
        predictions_image = draw_bounding_boxes(predictions_image, [instance.bboxes[j]], bbox_color = color_dict[lab], bbox_width = 2)


    acc = sum(instance.labels == predictions.prediction) / len(instance.labels)
    predictions_image = add_text_to_image(predictions_image, f"Accuracy {round(acc, 3)}")
    concat_images(labels_image, predictions_image).save(os.path.join(extraction_visualization_path, f"{id}.jpg"))
