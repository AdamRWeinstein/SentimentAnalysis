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
I've imported the IMDb review dataset from Keras which has pre-processed it for us. It provides each review as a sequence of word indices, the index of each word corresponding to its frequency across the dataset. We further process the data by padding and truncating it to the max length, which I decided should be 500 words, as that accomodates over 90% of the reviews in the dataset. We take these sequences and pass them into our RNN, which has the following architecture:
- Embedding Layer (Dimension of 50)
- LSTM Layer (64 Units)
- Dense Layer (1 node with a sigmoid activation function)

Training this network achieved the following results


|   Epoch |   Train_Accuracy |   Val_Accuracy |   Train_Precision |   Val_Precision |   Train_Recall |   Val_Recall |   Train_Loss |   Val_Loss |
|--------:|-----------------:|---------------:|------------------:|----------------:|---------------:|-------------:|-------------:|-----------:|
|       1 |           0.5042 |         0.5084 |            0.5052 |          0.5584 |         0.4088 |       0.0799 |       0.6947 |     0.6911 |
|       2 |           0.5187 |         0.5153 |            0.5250 |          0.6862 |         0.3916 |       0.0563 |       0.6825 |     0.6887 |
|       3 |           0.5350 |         0.5188 |            0.5377 |          0.7138 |         0.5001 |       0.0626 |       0.6640 |     0.6872 |
|       4 |           0.5387 |         0.5222 |            0.5427 |          0.7136 |         0.4920 |       0.0742 |       0.6470 |     0.6936 |
|       5 |           0.5479 |         0.5112 |            0.5606 |          0.5058 |         0.4429 |       0.9794 |       0.6441 |     0.7056 |
|       6 |           0.5380 |         0.5125 |            0.5490 |          0.5065 |         0.4262 |       0.9754 |       0.6373 |     0.7102 |
|       7 |           0.5439 |         0.5179 |            0.5514 |          0.7082 |         0.4713 |       0.0610 |       0.6519 |     0.6985 |
|       8 |           0.6459 |         0.7008 |            0.6387 |          0.6562 |         0.6721 |       0.8437 |       0.5946 |     0.6363 |
|       9 |           0.7355 |         0.7204 |            0.7083 |          0.7113 |         0.8006 |       0.7422 |        0.551 |     0.6092 |
|      10 |           0.7518 |         0.7202 |            0.7247 |          0.7002 |         0.8120 |       0.7701 |       0.5329 |     0.6112 |
|      11 |           0.6678 |         0.5216 |            0.6607 |          0.6992 |         0.6899 |       0.0757 |       0.5821 |     0.7000 |
|      12 |           0.5499 |         0.5248 |            0.5688 |          0.7281 |         0.4124 |       0.0793 |       0.6368 |     0.7048 |
|      13 |           0.5645 |         0.5073 |            0.5687 |          0.5038 |         0.5338 |       0.9646 |       0.6538 |     0.6909 |
|      14 |           0.6851 |         0.8410 |            0.6543 |          0.8344 |         0.7850 |       0.8510 |       0.5500 |     0.3882 |
|      15 |           0.8905 |         0.8655 |            0.8883 |          0.8389 |         0.8934 |       0.9046 |       0.2812 |     0.3206 |
|      16 |           0.9322 |         0.8732 |            0.9320 |          0.8994 |         0.9324 |       0.8404 |       0.1900 |     0.3287 |
|      17 |           0.9563 |         0.8708 |            0.9548 |          0.9043 |         0.9579 |       0.8294 |       0.1385 |     0.3623 |
|      18 |           0.9678 |         0.8712 |            0.9660 |          0.8642 |         0.9697 |       0.8809 |        0.106 |     0.3841 |
|      19 |           0.9801 |         0.8702 |            0.9786 |          0.8791 |         0.9817 |       0.8583 |       0.0771 |     0.4366 |
|      20 |           0.9850 |         0.8670 |            0.9839 |          0.8790 |         0.9861 |       0.8513 |       0.0643 |     0.4758 |

Notice how the training accuracy and the validation accuracy remain close to one another through the first 15 epochs, but begin to diverge with the training accuracy approaching 100% and the validation accuracy plateuing. This indicates that the model has overfit, so we can take the model saved at the end of the 15th Epoch as our model going forward. However, I would like to achieve a higher accuracy on unseen data, which means I will have to try either tuning the hyperparameters such as the learning rate, or adjusting steps with pre-processing steps such as the number of words we capture for our word indices or excluding some of the most common words. Alternatively, I can attempt new architecture for our RNN, such as adding stacked LSTM layers or introducing regularization.
