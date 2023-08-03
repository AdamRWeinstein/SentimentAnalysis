from preprocessing import preprocess_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from joblib import dump, load


# Function to transform text data
def transform_text(train_texts, test_texts):
    # Step 1: Convert text to BoW representation
    count_vectorizer = CountVectorizer(preprocessor=preprocess_text)
    train_bow = count_vectorizer.fit_transform(train_texts)
    test_bow = count_vectorizer.transform(test_texts)

    # Step 2: Convert BoW to TF-IDF representation
    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(train_bow)
    test_tfidf = tfidf_transformer.transform(test_bow)

    return train_tfidf, test_tfidf


# Function to build the Naive Bayes model
def build_model():
    model = MultinomialNB()
    return model


# Function to train the Naive Bayes model
def train_model(model, train_texts, train_labels):
    model.fit(train_texts, train_labels)
    return model


# Function to evaluate the Naive Bayes model
def evaluate_model(model, test_texts, test_labels):
    predictions = model.predict(test_texts)
    metrics = {
        'accuracy': accuracy_score(test_labels, predictions),
        'precision': precision_score(test_labels, predictions),
        'recall': recall_score(test_labels, predictions),
        'f1_score': f1_score(test_labels, predictions)
    }
    return metrics


# Function to save the model
def save_model(model, filename):
    dump(model, filename)


# Function to load the model
def load_model(filename):
    return load(filename)


# Function to find the best hyperparameters
def hyperparameter_tuning(model, train_texts, train_labels):
    # Define the hyperparameters and their possible values
    param_grid = {
        'alpha': [0.1, 0.5, 1.0]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(train_texts, train_labels)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score
