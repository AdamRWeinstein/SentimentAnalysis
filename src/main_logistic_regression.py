import os
import logistic_regression_model
from sklearn.utils import shuffle


def load_reviews(path):
    reviews = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
            reviews.append(file.read())
    return reviews


# Paths to the data directories
train_pos_path = '../data/aclImdb/train/pos'
train_neg_path = '../data/aclImdb/train/neg'
test_pos_path = '../data/aclImdb/test/pos'
test_neg_path = '../data/aclImdb/test/neg'

# Load the reviews
train_pos_reviews = load_reviews(train_pos_path)
train_neg_reviews = load_reviews(train_neg_path)
test_pos_reviews = load_reviews(test_pos_path)
test_neg_reviews = load_reviews(test_neg_path)

# Combine positive and negative reviews and create corresponding labels
train_reviews = train_pos_reviews + train_neg_reviews
train_labels = [1] * len(train_pos_reviews) + [0] * len(train_neg_reviews)
test_reviews = test_pos_reviews + test_neg_reviews
test_labels = [1] * len(test_pos_reviews) + [0] * len(test_neg_reviews)

# Shuffle the reviews and labels
train_reviews, train_labels = shuffle(train_reviews, train_labels, random_state=42)
test_reviews, test_labels = shuffle(test_reviews, test_labels, random_state=42)

# Build the model and transform the text data
model = logistic_regression_model.build_model()
train_reviews, test_reviews = logistic_regression_model.transform_text(train_reviews, test_reviews)

# Hyperparameter tuning
model, best_params, best_score = logistic_regression_model.hyperparameter_tuning(model, train_reviews, train_labels)
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)

# Evaluate the model
metrics = logistic_regression_model.evaluate_model(model, test_reviews, test_labels)

# Print metrics
print("Accuracy:", metrics['accuracy'])
print("Precision:", metrics['precision'])
print("Recall:", metrics['recall'])
print("F1 Score:", metrics['f1_score'])

# Save the model
logistic_regression_model.save_model(model, '../models/logistic_regression_model.pkl')
