from preprocessing import preprocess_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from joblib import dump, load


# Function to transform text data
def transform_text(train_texts, test_texts):
    # Create a pipeline for text transformation
    text_pipeline = Pipeline([
        ('bow', CountVectorizer(preprocessor=preprocess_text, ngram_range=(1, 1))),  # Convert text to BoW
        ('tfidf', TfidfTransformer()),  # Convert BoW to TF-IDF
        ('scaler', StandardScaler(with_mean=False))  # Apply standard scaling
    ])

    # Fit the pipeline on training data and transform both train and test data
    train_transformed = text_pipeline.fit_transform(train_texts)
    test_transformed = text_pipeline.transform(test_texts)

    return train_transformed, test_transformed


# Function to build the Logistic Regression model
def build_model():
    model = LogisticRegression(solver='saga', max_iter=5000, n_jobs=-1, penalty='elasticnet')
    return model


# Function to train the Logistic Regression model
def train_model(model, train_texts, train_labels):
    model.fit(train_texts, train_labels)
    return model


# Function to evaluate the Logistic Regression model
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
def hyperparameter_tuning(model, test_texts, test_labels):
    param_grid = {
        'C': [0.08, 0.1, 0.12],
        'l1_ratio': [0.5, 0.6, 0.7]
    }
    print("Starting hyperparameter tuning...")
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=10)
    grid_search.fit(test_texts, test_labels)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Hyperparameter tuning completed.")
    return best_model, best_params, best_score
