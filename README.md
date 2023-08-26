# Sentiment Analysis on IMDb Movie Reviews

## Overview

This project is focused on performing sentiment analysis on a dataset of IMDb movie reviews. The goal is to classify reviews as positive or negative based on the textual content. Three different models are planned to compare their performance on this task:

1. **Naive Bayes Classifier** ☑
2. **Logistic Regression** ☑
3. **Long Short-Term Memory Networks (LSTM)** ☐

The dataset consists of 50,000 movie reviews from the IMDb website, with 25,000 in the training set and 25,000 in the test set. The data was obtained from: https://ai.stanford.edu/~amaas/data/sentiment/

## Dependencies

- Python 3.11
- scikit-learn
- NLTK
- TensorFlow and Keras
- numpy

## Preprocessing

The preprocessing step involves several text cleaning and transformation tasks to prepare the data for machine learning models. All of the steps available include:

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

Each shallow learning model is run through a method for hyperparameter tuning to identify the best parameters for that model. The RNN is actively being explored with different configurations and regularization options.

### Naive Bayes and Logistic Regression

Before implementing a deep learning solution, I wanted to explore shallow learning solutions to familiarize myself as well as set a baseline for what to expect performance-wise. I've implemented both a Naive Bayes and a Logistic Regression model from the SciKit Learn library. Our approach is as follows:

- **Vectorization**: The text is converted into a numeric form using the Bag of Words (BoW) technique.
- **TF-IDF Transformation**: The BoW vectors are further transformed using Term Frequency - Inverse Document Frequency (TF-IDF) to weigh terms based on their significance.

Potential enhancements involve utilizing the `ngram_range` parameter in `CountVectorizer` to capture multi-word contexts. For example, adjusting the `ngram_range` to (1, 2) allows the model to consider both single words and two-word phrases (e.g. "I like words" with an ngram_range of (1, 2) would provide ["I", "like", "words", "I like", "like words"] rather than the current implementation with an ngram_range of (1, 1) providing ["I", "like", "words"]). This approach can help mitigate the model's assumption of token independence by incorporating local context directly into the tokenized data.


### LSTM

LSTM (Long Short-Term Memory) networks are a type of recurrent neural network (RNN) capable of learning long-term dependencies, making them suitable for sequence prediction problems like text classification. They are well suited for learning long-term dependencies in data as well as processing data sequentially, which makes them better suited for NLP tasks than models like Naive Bayes and Logistic Regression due to NB's assumption of independence and Logistic Regression's inability to capture sequential dependencies in data.

## Results

The performance of the models is evaluated using the following metrics: 

- **Accuracy**: The percentage of our predictions that are correct.
- **Precision**: Out of all the positive predictions we made, how many were actually positive.
- **Recall**: Out of all the positive instances in the dataset, how many did we correctly classify?
- **F1-score***: Combines both precision and recall, providing a single score showing the balance of both. This metric ensures that the model does not neglect either of these metrics.

* The F1-score metric is not readily available through the Keras library so I do not capture it for the RNN models.

The results indicate the strengths and weaknesses of each approach on the given dataset.

### Naive Bayes

- Best Parameters: {'alpha': 0.5}
- Accuracy: 0.8305
- Precision: 0.8636
- Recall: 0.7850
- F1 Score: 0.8224

### Logistic Regression

- Best Parameters: {'C': 0.1, 'l1_ratio': 0.6}
- Accuracy: 0.8622
- Precision: 0.8694
- Recall: 0.8523
- F1 Score: 0.8608

### RNN
To begin, I tested three different architectures utilizing LSTM units.
1. An RNN with a single LSTM layer with 64 units.
2. An RNN with stacked LSTM layers, two layers with 64 units each but has no regularization.
3. An RNN with stacked LSTM layers, two layers with 64 units each as well as 20% dropout and an l2 regularization of .001.

The initial results were as follows:

| Model Type                     | Data Split  | Accuracy | Precision | Recall  |
|--------------------------------|-------------|----------|-----------|---------|
| Single LSTM Layer              | Training    | 0.9104   | 0.9100    | 0.9110  |
|                                | Validation  | 0.6256   | 0.6267    | 0.6214  |
|                                | Test        | 0.6256   | 0.6267    | 0.6214  |
| Stacked LSTM (No Reg.)         | Training    | 0.9842   | 0.9937    | 0.9746  |
|                                | Validation  | 0.7572   | 0.7901    | 0.7004  |
|                                | Test        | 0.7572   | 0.7901    | 0.7004  |
| Stacked LSTM (With Reg.)       | Training    | 0.4914   | 0.4830    | 0.2461  |
|                                | Validation  | 0.5000   | 0.0000    | 0.0001  |
|                                | Test        | 0.5000   | 0.0000    | 0.0000  |

Based on these results, the power of an RNN is displayed in the high accuracy, precision, and recall achieved on the training set by the stacked RNN, however the problem of overfitting must be addressed. My initial attempt to regularize the model failed, as seen in the above results. Likely, the l2 regularization was too strong, preventing the model from learning any patterns in the data and essentially reducing it to random guessing. 
