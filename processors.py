# Author: Sergej Levich
# Journal article: Sergej Levich and Lucas Knust, International Journal of Accounting Information Systems, https://doi.org/10.1016/j.accinf.2025.100750

import numpy as np
from doc2data.utils import load_image
from doc2data.experimental.base_processors import DataProcessor
from doc2data.experimental.utils import to_categorical

class SequenceClsProcessorLayoutXLM(DataProcessor):
    """Processor for sequence classification (LayoutXLM fine-tuning)."""

    def __init__(
        self,
        instances_df,
        processor,
        pdf_collection
    ):
        super().__init__(instances_df, processor)
        self.pdf_collection = pdf_collection
        unique_labels = np.unique(instances_df.label)
        self.id2label = {v: k for v, k in enumerate(unique_labels)}
        self.label2id = {k: v for v, k in enumerate(unique_labels)}

    @staticmethod
    def normalize_bbox_layoutxlm_old(bbox, width, height):
         return [
             int(1000 * (bbox[0] / width)),
             int(1000 * (bbox[1] / height)),
             int(1000 * (bbox[2] / width)),
             int(1000 * (bbox[3] / height)),
         ]

    @staticmethod
    def normalize_bbox_layoutxlm(bbox, width, height):
         return [
             int(1000 * min(1, max(0, bbox[0]))),
             int(1000 * min(1, max(0, bbox[1]))),
             int(1000 * min(1, max(0, bbox[2]))),
             int(1000 * min(1, max(0, bbox[3]))),
         ]

    def load_instance(self, source):
        pdf_page = self.pdf_collection.pdfs[source.file_name][source.page_nr]
        image = pdf_page.read_contents('page_image', force_rgb = True)
        tokens = source.tokens
        bboxes = source.bboxes
        bboxes = [self.normalize_bbox_layoutxlm(i, image.width, image.height) for i in bboxes]
        label = self.label2id[source.label]

        return (image, tokens, bboxes), label

    def get_processed_instance(self, source):
        features, label = self.load_instance(source)
        if self.processor:
            instance = self.processor(features, label, self.inference_mode)
        else:
            instance = (features, label)

        return instance

    def __getitem__(self, idx):
        row = self.instances_df.iloc[idx]
        instance = self.get_processed_instance(row)

        return instance


class TokenClsProcessorLayoutXLM(DataProcessor):
    """Processor for token classification (LayoutXLM fine-tuning)."""

    def __init__(
        self,
        instances_df,
        processor,
        pdf_collection
    ):
        super().__init__(instances_df, processor)
        self.pdf_collection = pdf_collection
        unique_labels = np.unique([i for j in instances_df.labels for i in j])
        self.id2label = {v: k for v, k in enumerate(unique_labels)}
        self.label2id = {k: v for v, k in enumerate(unique_labels)}

    @staticmethod
    def normalize_bbox_layoutxlm_old(bbox, width, height):
         return [
             int(1000 * (bbox[0] / width)),
             int(1000 * (bbox[1] / height)),
             int(1000 * (bbox[2] / width)),
             int(1000 * (bbox[3] / height)),
         ]

    @staticmethod
    def normalize_bbox_layoutxlm(bbox, width, height):
         return [
             int(1000 * min(1, max(0, bbox[0]))),
             int(1000 * min(1, max(0, bbox[1]))),
             int(1000 * min(1, max(0, bbox[2]))),
             int(1000 * min(1, max(0, bbox[3]))),
         ]

    def load_instance(self, source):
        pdf_page = self.pdf_collection.pdfs[source.file_name][source.page_nr]
        image = pdf_page.read_contents('page_image', force_rgb = True)
        tokens = source.tokens
        bboxes = source.bboxes
        bboxes = [self.normalize_bbox_layoutxlm(i, image.width, image.height) for i in bboxes]
        labels = [self.label2id[i] for i in source.labels]

        return (image, tokens, bboxes), labels

    def get_processed_instance(self, source):
        features, labels = self.load_instance(source)
        if self.processor:
            instance = self.processor(features, labels, self.inference_mode)
        else:
            instance = (features, labels)

        return instance

    def __getitem__(self, idx):
        row = self.instances_df.iloc[idx]
        instance = self.get_processed_instance(row)

        return instance
