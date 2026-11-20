# Naive Bayes Sentiment Analysis from Scratch

Developed a binary sentiment classification system using a **Naive Bayes Classifier implemented entirely in Python**, without relying on external machine learning or NLP libraries.

The project classifies customer reviews as positive or negative based on their star ratings, using natural language processing techniques, feature engineering, and probabilistic classification.

## Key Features

- Implemented a Naive Bayes classifier from scratch using Python.
- Developed text preprocessing with tokenization, lowercase conversion, and stop-word filtering.
- Designed a feature-selection pipeline using the 700 most frequent unigrams and 300 most frequent bigrams.
- Applied Laplace smoothing to estimate conditional feature probabilities.
- Used logarithmic probabilities for numerical stability during classification.
- Evaluated classification performance using precision, recall, and F1 score.
- Analyzed misclassified reviews to identify limitations in sentiment prediction.
- Documented learning curve experiments and vocabulary ablation comparisons.

## Model Performance

| Metric | Score |
|---|---|
| Precision | 0.8857 |
| Recall | 0.8623 |
| F1 Score | 0.8738 |

The classifier achieved an **F1 score of 87.38%** on the test dataset.

## Experimental Analysis

The project report also investigates:

- **Learning Curve:** Evaluated how training dataset size affects model performance.
- **Vocabulary Ablation:** Compared unigram-only, bigram-only, and combined feature configurations.
- **Error Analysis:** Examined incorrect predictions to identify challenges with context and sentiment interpretation.

## Technologies

Python · Natural Language Processing (NLP) · Naive Bayes · Text Classification · Feature Engineering · Probabilistic Machine Learning

**Libraries:** Python Standard Library (`csv`, `re`, `math`, `collections`)
