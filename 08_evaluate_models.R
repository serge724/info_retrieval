library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)
library(magrittr)
library(caret)

options(pillar.sigfig = 4)

# page identification
# load training history
page_ident_training <- read_csv('results/page_identification/layoutxlm/training_history.csv') %>% 
  rename(epoch = `...1`)

page_ident_training %>% 
  gather(metric, value, -epoch, -data_split) %>% 
  mutate(
    epoch = epoch + 1,
    data_split = ifelse(data_split == "train_set", "Training", "Validation"),
    metric = ifelse(metric == "accuracy", "Accuracy", "Average Loss")
  ) %>% 
  ggplot(aes(epoch, value, color = data_split)) +
  geom_line() +
  facet_wrap(~metric, nrow = 1, scales = 'free') +
  theme_bw() +
  ylab("") +
  xlab("Epoch") + 
  theme(legend.title=element_blank()) +
  viridis::scale_color_viridis(discrete = TRUE)

# load layoutxlm test set predictions
page_ident <- read_csv('results/page_identification/layoutxlm/test_set_predictions.csv') %>% 
  mutate(label = as.factor(as.integer(label)), prediction = as.factor(prediction)) %>% 
  select(-company) %>% 
  left_join(
    read_csv('data/labels/page_identification_labels.csv') %>% 
      janitor::clean_names() %>% 
      distinct(file_name, language, accounting_rules, company, no_subs, foreign_issuer),
    by = 'file_name'
  )

# show metrics
caret::confusionMatrix(page_ident$prediction, page_ident$label, positive = '1')
caret::confusionMatrix(page_ident$prediction, page_ident$label, positive = '1')$byClass %>% tibble::enframe()

# precision, recall & f1 tradeoffs
threshold_tradeoff <- tibble(NULL)
for (threshold in rev(seq(0.05, 0.95, 0.05))) {
  adjusted <- page_ident %>%
    mutate(prediction = as.factor(ifelse(probability > threshold, 1, 0)))
  
  accuracy <- caret::confusionMatrix(adjusted$prediction, adjusted$label, positive = '1')$overall['Accuracy']
  precision <- caret::confusionMatrix(adjusted$prediction, adjusted$label, positive = '1')$byClass['Precision']
  recall <- caret::confusionMatrix(adjusted$prediction, adjusted$label, positive = '1')$byClass['Recall']
  f1 <- caret::confusionMatrix(adjusted$prediction, adjusted$label, positive = '1')$byClass['F1']
  threshold_tradeoff %<>%
    bind_rows(
      tibble(threshold = threshold, accuracy, precision = precision, recall = recall, f1 = f1)
    )
}

threshold_tradeoff %>% 
  arrange(-threshold) %>% 
  transmute(Precision = precision, Recall = recall, Threshold = scales::percent(threshold, scale = 100)) %>% 
  ggplot(aes(Recall, Precision)) +
  # geom_step(direction = "hv") +
  geom_point() +
  geom_text(aes(label = Threshold), nudge_y = 0.01, nudge_x = 0.01, size = 3) +
  theme_bw() +
  scale_y_continuous(labels = scales::percent) +
  scale_x_continuous(labels = scales::percent) +
  viridis::scale_color_viridis(discrete = TRUE)

threshold_tradeoff %>% 
  arrange(-threshold) %>% 
  transmute(Threshold = threshold, F1 = f1) %>% 
  ggplot(aes(Threshold, F1)) +
  # geom_step(direction = "hv") +
  geom_point() +
  theme_bw() +
  scale_y_continuous(labels = scales::percent, limits = c(0, NA)) +
  scale_x_continuous(labels = scales::percent, breaks = scales::pretty_breaks(n = 10)) +
  viridis::scale_color_viridis(discrete = TRUE)

# show metrics by language
page_ident %>% 
  group_by(language) %>% 
  summarise(
    accuracy = caret::confusionMatrix(prediction, label, positive = '1')$overall["Accuracy"],
    precision = caret::confusionMatrix(prediction, label, positive = '1')$byClass["Precision"],
    recall = caret::confusionMatrix(prediction, label, positive = '1')$byClass["Recall"],
    f1_score = caret::confusionMatrix(prediction, label, positive = '1')$byClass["F1"],
    support = n()
  ) %>% 
  arrange(-support)

# show metrics by accounting rules
page_ident %>% 
  group_by(accounting_rules) %>% 
  summarise(
    accuracy = caret::confusionMatrix(prediction, label, positive = '1')$overall["Accuracy"],
    precision = caret::confusionMatrix(prediction, label, positive = '1')$byClass["Precision"],
    recall = caret::confusionMatrix(prediction, label, positive = '1')$byClass["Recall"],
    f1_score = caret::confusionMatrix(prediction, label, positive = '1')$byClass["F1"],
    support = n()
  ) %>% 
  arrange(-support)

# load gpt-4o validation set predictions
llm_page_ident_val_set <- read_csv('results/page_identification/chatgpt/llm_predictions_val_set.csv') %>% 
  mutate(label = as.factor(as.integer(label)), prediction = as.factor(as.integer(prediction)))

# show metrics
llm_page_ident_val_set %>% 
  group_by(prompt_id) %>% 
  summarise(
    accuracy = caret::confusionMatrix(prediction, label, positive = '1')$overall["Accuracy"],
    precision = caret::confusionMatrix(prediction, label, positive = '1')$byClass["Precision"],
    recall = caret::confusionMatrix(prediction, label, positive = '1')$byClass["Recall"],
    f1_score = caret::confusionMatrix(prediction, label, positive = '1')$byClass["F1"],
    support = n()
  )

# show confusion matrices
for (i in unique(llm_page_ident_val_set$prompt_id)) {
  prompt_predictions <- llm_page_ident_val_set %>% 
    filter(prompt_id == i)
  caret::confusionMatrix(prompt_predictions$prediction, prompt_predictions$label, positive = '1')$table %>% 
    print()
}

# load gpt-4o validation test predictions
llm_page_ident_test_set <- read_csv('results/page_identification/chatgpt/llm_predictions_test_set.csv') %>% 
  mutate(label = as.factor(as.integer(label)), prediction = as.factor(as.integer(prediction))) %>% 
  select(-company) %>% 
  left_join(
    read_csv('data/labels/page_identification_labels.csv') %>% 
      distinct(file_name, language, accounting_rules, company, no_subs, foreign_issuer),
    by = 'file_name'
  )

# show metrics
caret::confusionMatrix(llm_page_ident_test_set$prediction, llm_page_ident_test_set$label, positive = '1')
caret::confusionMatrix(llm_page_ident_test_set$prediction, llm_page_ident_test_set$label, positive = '1')$byClass %>% tibble::enframe()

# show metrics by language
llm_page_ident_test_set %>% 
  group_by(prompt_id, language) %>% 
  summarise(
    accuracy = caret::confusionMatrix(prediction, label, positive = '1')$overall["Accuracy"],
    precision = caret::confusionMatrix(prediction, label, positive = '1')$byClass["Precision"],
    recall = caret::confusionMatrix(prediction, label, positive = '1')$byClass["Recall"],
    f1_score = caret::confusionMatrix(prediction, label, positive = '1')$byClass["F1"],
    support = n()
  ) %>% 
  arrange(-support)

# show metrics by accounting rules
llm_page_ident_test_set %>% 
  group_by(prompt_id, accounting_rules) %>% 
  summarise(
    accuracy = caret::confusionMatrix(prediction, label, positive = '1')$overall["Accuracy"],
    precision = caret::confusionMatrix(prediction, label, positive = '1')$byClass["Precision"],
    recall = caret::confusionMatrix(prediction, label, positive = '1')$byClass["Recall"],
    f1_score = caret::confusionMatrix(prediction, label, positive = '1')$byClass["F1"],
    support = n()
  ) %>% 
  arrange(-support)

# information extraction
# load training history
info_extra_training <- read_csv('results/information_extraction/layoutxlm/training_history.csv') %>% 
  rename(epoch = `...1`)

info_extra_training %>% 
  gather(metric, value, -epoch, -data_split) %>% 
  mutate(
    epoch = epoch + 1,
    data_split = ifelse(data_split == "train_set", "Training", "Validation"),
    metric = ifelse(metric == "accuracy", "Accuracy", "Average Loss")
  ) %>% 
  ggplot(aes(epoch, value, color = data_split)) +
  geom_line() +
  facet_wrap(~metric, nrow = 1, scales = 'free') +
  theme_bw() +
  ylab("") +
  xlab("Epoch") + 
  theme(legend.title=element_blank()) +
  viridis::scale_color_viridis(discrete = TRUE)

# load layoutxlm test set predictions
info_extra <- read_csv('results/information_extraction/layoutxlm/test_set_predictions.csv') %>% 
  mutate(label = as.factor(label), prediction = as.factor(prediction))
info_extra %<>% 
  left_join(
    read_csv('data/labels/page_identification_labels.csv') %>% 
      distinct(file_name, language, accounting_rules, company, no_subs, foreign_issuer),
    by = 'file_name'
  )

# show metrics
caret::confusionMatrix(info_extra$prediction, info_extra$label)
caret::confusionMatrix(info_extra$prediction, info_extra$label)$byClass[, c('Precision', 'Recall', 'F1')]

info_extra %>% 
  summarise(
    metrics = list(confusionMatrix(prediction, label)$byClass),
    accuracy = confusionMatrix(prediction, label)$overall['Accuracy'],
    n = n(),
    .groups = 'drop'
  ) %>% 
  mutate(metrics = map(metrics, as_tibble, rownames = 'type')) %>% 
  unnest(metrics) %>% 
  janitor::clean_names() %>% 
  transmute(type = str_remove(type, 'Class: '), prevalence, precision, recall, f1, n, accuracy) %>% 
  summarise(
    n = unique(n),
    accuracy = unique(accuracy),
    precision_macro = mean(precision),
    recall_macro = mean(recall),
    f1_macro = mean(f1)
  )

# show metrics by language
info_extra %>% 
  group_by(language) %>% 
  summarise(
    metrics = list(confusionMatrix(prediction, label)$byClass),
    accuracy = confusionMatrix(prediction, label)$overall['Accuracy'],
    n = n(),
    .groups = 'drop'
  ) %>% 
  mutate(metrics = map(metrics, as_tibble, rownames = 'type')) %>% 
  unnest(metrics) %>% 
  janitor::clean_names() %>% 
  transmute(language, type = str_remove(type, 'Class: '), prevalence, precision, recall, f1, n, accuracy) %>% 
  group_by(language) %>% 
  summarise(
    n = unique(n),
    accuracy = unique(accuracy),
    precision_macro = mean(precision),
    recall_macro = mean(recall),
    f1_macro = mean(f1)
  )

# show metrics by accounting rules
info_extra %>% 
  group_by(accounting_rules) %>% 
  summarise(
    metrics = list(confusionMatrix(prediction, label)$byClass),
    accuracy = confusionMatrix(prediction, label)$overall['Accuracy'],
    n = n(),
    .groups = 'drop'
  ) %>% 
  mutate(metrics = map(metrics, as_tibble, rownames = 'type')) %>% 
  unnest(metrics) %>% 
  janitor::clean_names() %>% 
  transmute(accounting_rules, type = str_remove(type, 'Class: '), prevalence, precision, recall, f1, n, accuracy) %>% 
  group_by(accounting_rules) %>% 
  summarise(
    n = unique(n),
    accuracy = unique(accuracy),
    precision_macro = mean(precision),
    recall_marco = mean(recall),
    f1_macro = mean(f1)
  )