# About
This repository contains the code for the article "Discriminative meets generative: Automated information retrieval from unstructured corporate documents via (large) language models" published in International Journal of Accounting Information Systems.

**Article DOI: [10.1016/j.accinf.2025.100750](https://doi.org/10.1016/j.accinf.2025.100750)**

**Dataset DOI: [10.17632/39kyzcp9r6.1](https://doi.org/10.17632/39kyzcp9r6.1)**

The code is set up as a collection of scripts that need to be executed in sequential order.

* **00_create_data_split.R:** Creates a dataset split based on the issuers to prevent data leakage.
* **01_parse_pdfs.py:** Uses the library `doc2data` to parse PDF documents and create a collection.
* **02_create_instances.py:** Creates instances for both tasks, namely page identification and information extraction. For training and testing the discriminative model, text and layout features are extracted. For engineering and testing the prompts for the generative model, only text is used.
* **03_train_test_identification.py:** Fine-tunes the LayoutXLM model for page identification and evaluates the best checkpoint on the test set.
* **04_train_test_extraction.py:**  Fine-tunes the LayoutXLM model for information extraction and evaluates the best checkpoint on the test set.
* **05_run_llm_identification.py:** Runs the query pipeline using GPT-4o for page identification with candidate prompts on the validation set and the best prompt on the test set.
* **06_run_llm_extraction.py:** Runs the query pipeline using GPT-4o for information extraction with candidate prompts on the validation set and the best prompt on the test set.
* **07_parse_llm_predictions.py:** Parses LLM predictions (including error handling) and creates CSV files for evaluation.
* **08_evaluate_predictions.R:** Calculates all performance metrics as reported in the paper.
* **09_visualize_extraction_predictions.py:** Generates images with labels and predictions on word bounding boxes for the information extraction task for visual inspection.

# Usage
This code was developed and executed on a system running Ubuntu 22.04, equipped with 64GB of RAM and an Nvidia RTX4090 GPU with 24GB of VRAM. To reproduce the results please follow the outlined steps:

## 1. Clone repo
```
git clone https://github.com/serge724/info_retrieval.git
cd info_retrieval
```

## 2. Download datasets
You can access and download the required datasets from [Mendeley Data](https://data.mendeley.com/datasets/39kyzcp9r6/1).

Ensure that the files are placed as follows:
* data/annual_reports/
* data/labels/bbox_labels.json
* data/labels/page_identification_labels.csv
* data/data_split.csv

## 3. Create conda environment
```
conda create -n info_retrieval python=3.11.7
conda activate info_retrieval
```

## 4. Install Python packages
```
conda install pytorch=2.3.1 torchvision=0.18.1 pytorch-cuda=12.1 pillow=9 -c pytorch -c nvidia
pip install doc2data==0.2.0 PyMuPDF==1.24.7 transformers==4.41.2 "openai>=1.35.13,<2.0" scikit-learn==1.5.1 evaluate==0.4.2 matplotlib==3.9.0 tqdm ipykernel ipywidgets
pip install git+https://github.com/facebookresearch/detectron2.git@v0.6
```

## 5. Install R libraries
The R scripts rely on the following libraries:
```
readr 2.1.5
dplyr 1.1.4
tidyr 1.3.1
stringr 1.5.1
purrr 1.0.4
magrittr 2.0.3
ggplot2 3.5.2
caret 7.0-1
```
While the R scripts may work with different versions of the listed libraries, this was not tested.

## 6. Run scripts

In order to reproduce our results, you can run the scripts sequentially from the terminal:

```
Rscript 00_create_data_split.R
python 01_parse_pdfs.py
python 02_create_instances.py
python 03_train_test_identification.py
python 04_train_test_extraction.py
python 05_run_llm_identification.py
python 06_run_llm_extraction.py
python 07_parse_llm_predictions.py
Rscript 08_evaluate_models.R
python 09_visualize_extraction_predictions.py
```

Processed instances are saved to the directory ```processed_data/```, model checkpoints to ```models/``` and predictions as well as visualizations are saved to ```results/```. Please note that random initialization and a different computational environment may result in slight deviations.

# Citation
If you find this code useful, please consider citing our work as follows:

```BibTeX
@article{LEVICH2025100750,
    title = {Discriminative meets generative: Automated information retrieval from unstructured corporate documents via (large) language models},
    journal = {International Journal of Accounting Information Systems},
    volume = {56},
    pages = {100750},
    year = {2025},
    issn = {1467-0895},
    doi = {https://doi.org/10.1016/j.accinf.2025.100750},
    url = {https://www.sciencedirect.com/science/article/pii/S1467089525000260},
    author = {Sergej Levich and Lucas Knust}
}
```