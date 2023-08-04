# Sentiment Analysis on IMDb Movie Reviews

## Overview

This project is focused on performing sentiment analysis on a dataset of IMDb movie reviews. The goal is to classify reviews as positive or negative based on the textual content. Three different models are planned to compare their performance on this task:

1. **Naive Bayes Classifier** ☑
2. **Logistic Regression** ☐
3. **Long Short-Term Memory Networks (LSTM)** ☐

The dataset consists of 50,000 movie reviews from the IMDb website, with 25,000 in the training set and 25,000 in the test set. The data was obtained from: https://ai.stanford.edu/~amaas/data/sentiment/

## Dependencies

- Python 3.11
- scikit-learn
- NLTK
- TensorFlow (for LSTM)
- contractions
- Other standard data science libraries

## Preprocessing

The preprocessing step involves several text cleaning and transformation tasks, including:

- Removal of HTML tags.
- Replacement of specific ratings with general labels (positive, neutral, negative).
- Removal of any remaining numbers
- Expansion of contractions.
- Conversion to lowercase.
- Lemmatization.
- Removal of stop words.
- Removal or replacement of punctuation.
- Removal of excess whitespace.

## Models

Each model is run through a method for hyperparameter tuning to identify the best parameters for that model.

### Naive Bayes

The Naive Bayes model is a probabilistic classifier based on applying Bayes' theorem. It's a simple yet effective model for text classification tasks.

### Logistic Regression

Logistic Regression is used to model the probability of a binary outcome. It’s a statistical method for analyzing datasets and works well for binary classification.

### LSTM

LSTM (Long Short-Term Memory) networks are a type of recurrent neural network (RNN) capable of learning long-term dependencies, making them suitable for sequence prediction problems like text classification.

## Results

The performance of the models is evaluated using the following metrics: accuracy, precision, recall, and F1-score. The results indicate the strengths and weaknesses of each approach on the given dataset.

## Naive Bayes

Best Parameters: {'alpha': 0.5}
Best Cross-Validation Score: 0.8699999999999999
Accuracy: 0.83052
Precision: 0.8635923611722256
Recall: 0.78504
F1 Score: 0.8224447890038973
