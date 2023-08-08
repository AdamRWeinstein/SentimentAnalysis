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

The preprocessing step involves several text cleaning and transformation tasks to prepare the data for machine learning models. The steps include:

- **Removal of HTML tags**: Ensures that the model isn't influenced by any web markup.
- **Rating Replacement**: Specific ratings are replaced with general labels (positive, neutral, negative) to provide a uniform representation.
- **Number Removal**: Any standalone numbers that don't represent ratings are removed.
- **Expansion of Contractions**: Expands contractions (e.g., "isn't" becomes "is not") for uniformity.
- **Punctuation Handling**: Punctuation is either removed or replaced with spaces.
- **Case Normalization**: Converts all text to lowercase to avoid duplication based on case differences.
- **Lemmatization**: Words are reduced to their base or dictionary form (e.g. 'better' -> 'good')
- **Stop Word Removal**: Common words that don't carry significant meaning are removed.
- **Whitespace Handling**: Any excess white spaces are removed.

Potential enhancements for preprocessing include refining the rating categorization strategy, such as omitting neutral placeholders for explicit ratings between 40-60% and categorizing them as either positive or negative. Another option is to discard neutral ratings before tokenization.

## Models

Each model is run through a method for hyperparameter tuning to identify the best parameters for that model.

### Naive Bayes

The Naive Bayes model is a probabilistic classifier that applies Bayes' theorem with strong (naive) independence assumptions. Despite its simplicity, it's often effective for text classification tasks. Our approach involves:

- **Vectorization**: The text is converted into a numeric form using the Bag of Words (BoW) technique.
- **TF-IDF Transformation**: The BoW vectors are further transformed using Term Frequency - Inverse Document Frequency (TF-IDF) to weigh terms based on their significance.

Potential enhancements involve utilizing the `ngram_range` parameter in `CountVectorizer` to capture multi-word contexts. For example, adjusting the `ngram_range` to (1, 2) allows the model to consider both single words and two-word phrases (e.g. ("I like words" with an ngram_range of (1, 2) would provide ["I", "like", "words", "I like", "like words"] rather than the current implementation with an ngram_range of (1, 1) providing ["I", "like", "words"]). This approach can help mitigate the model's assumption of token independence by incorporating local context directly into the tokenized data.


### Logistic Regression

Logistic Regression is used to model the probability of a binary outcome. It’s a statistical method for analyzing datasets and works well for binary classification.

### LSTM

LSTM (Long Short-Term Memory) networks are a type of recurrent neural network (RNN) capable of learning long-term dependencies, making them suitable for sequence prediction problems like text classification.

## Results

The performance of the models is evaluated using the following metrics: accuracy, precision, recall, and F1-score. The results indicate the strengths and weaknesses of each approach on the given dataset.

### Naive Bayes

Best Parameters: {'alpha': 0.5}

Best Cross-Validation Score: 0.8699999999999999

Accuracy: 0.83052

Precision: 0.8635923611722256

Recall: 0.78504

F1 Score: 0.8224447890038973
