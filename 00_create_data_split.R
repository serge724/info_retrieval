library(readr)
library(dplyr)
library(magrittr)

# read page identification labels
identification_labels <- read_csv('data/labels/page_identification_labels.csv')

# remove foreign issuers
identification_labels %<>% 
  filter(!foreign_issuer)

# checks
identification_labels %>% 
  count(is.na(page_label))
identification_labels %>% 
  distinct(file_name, no_subs) %>%
  count(no_subs)

companies <- identification_labels %>% 
  distinct(company)

# add split labels according to a 80%/10%/10% split
companies$data_split <- 'test_set'
companies[1:floor(nrow(companies) * 0.9),]$data_split <- 'val_set'  
companies[1:floor(nrow(companies) * 0.8),]$data_split <- 'train_set'  

companies %>% 
  count(data_split)

# shuffle dataset
set.seed(123)
companies %<>% 
  mutate(company = sample(company, length(company)))

# write split labels to disk
companies %>%
  write_csv('data/data_split.csv')