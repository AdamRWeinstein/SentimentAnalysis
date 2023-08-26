import os
from sklearn.utils import shuffle
from preprocessing import preprocess_text_RNN
from rnn_model import create_rnn_model, create_rnn_model_stacked, create_rnn_model_stacked_regularized
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.backend import clear_session
import numpy as np


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

# # Shuffle the reviews and labels
train_reviews, train_labels = shuffle(train_reviews, train_labels)
test_reviews, test_labels = shuffle(test_reviews, test_labels)

# Preprocess data
train_reviews = [preprocess_text_RNN(review) for review in train_reviews]
test_reviews = [preprocess_text_RNN(review) for review in test_reviews]

# Tokenization and Padding
MAX_WORDS = 60000
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<UNK>')
tokenizer.fit_on_texts(train_reviews)

train_sequences = tokenizer.texts_to_sequences(train_reviews)
test_sequences = tokenizer.texts_to_sequences(test_reviews)

MAX_LENGTH = 1000
train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# RNN Model with a single LSTM Layer
# Define the Learning Rate Schedule
def lr_schedule(epoch):
    if epoch < 5:
        return 0.01
    else:
        return 0.001


# Set up the callbacks
scheduler = LearningRateScheduler(lr_schedule)
model_checkpoint = ModelCheckpoint('../models/single_rnn_model_epoch_{epoch:02d}.keras',
                                   save_best_only=False,
                                   save_weights_only=False)
callbacks = [model_checkpoint, scheduler]

# Instantiate and train the model
model = create_rnn_model(MAX_WORDS, MAX_LENGTH)
history = model.fit(train_padded, train_labels, epochs=30, batch_size=64, validation_data=(test_padded, test_labels), verbose=1, callbacks=callbacks)

# Parse out the metrics
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_precision = history.history['precision']
val_precision = history.history['val_precision']
train_recall = history.history['recall']
val_recall = history.history['val_recall']

print(f"Training Accuracy: {train_accuracy[-1]:.4f}")
print(f"Validation Accuracy: {val_accuracy[-1]:.4f}")
print(f"Training Precision: {train_precision[-1]:.4f}")
print(f"Validation Precision: {val_precision[-1]:.4f}")
print(f"Training Recall: {train_recall[-1]:.4f}")
print(f"Validation Recall: {val_recall[-1]:.4f}")

loss, accuracy, precision, recall = model.evaluate(test_padded, test_labels, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")

model.save('../models/single_rnn_model.keras')
clear_session()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# RNN Model with two LSTM layers

# Callbacks
model_checkpoint = ModelCheckpoint('../models/stacked_rnn_model_epoch_{epoch:02d}.keras',
                                   save_best_only=False,
                                   save_weights_only=False)
callbacks = [model_checkpoint, scheduler]

# Instantiate and train the model
model = create_rnn_model_stacked(MAX_WORDS, MAX_LENGTH)
history = model.fit(train_padded, train_labels, epochs=30, batch_size=64, validation_data=(test_padded, test_labels), verbose=1, callbacks=callbacks)

# Parse out the metrics
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_precision = history.history['precision']
val_precision = history.history['val_precision']
train_recall = history.history['recall']
val_recall = history.history['val_recall']

print(f"Training Accuracy: {train_accuracy[-1]:.4f}")
print(f"Validation Accuracy: {val_accuracy[-1]:.4f}")
print(f"Training Precision: {train_precision[-1]:.4f}")
print(f"Validation Precision: {val_precision[-1]:.4f}")
print(f"Training Recall: {train_recall[-1]:.4f}")
print(f"Validation Recall: {val_recall[-1]:.4f}")

loss, accuracy, precision, recall = model.evaluate(test_padded, test_labels, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")


model.save('../models/stacked_rnn_model.keras')
clear_session()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# RNN Model with two LSTM layers and regularization
model_checkpoint = ModelCheckpoint('../models/regularized_rnn_model_epoch_{epoch:02d}.keras',
                                   save_best_only=False,
                                   save_weights_only=False)
callbacks = [model_checkpoint, scheduler]

# Instantiate and train the model
model = create_rnn_model_stacked_regularized(MAX_WORDS, MAX_LENGTH)
history = model.fit(train_padded, train_labels, epochs=30, batch_size=64, validation_data=(test_padded, test_labels), verbose=1, callbacks=callbacks)

# Parse out the metrics
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_precision = history.history['precision']
val_precision = history.history['val_precision']
train_recall = history.history['recall']
val_recall = history.history['val_recall']

print(f"Training Accuracy: {train_accuracy[-1]:.4f}")
print(f"Validation Accuracy: {val_accuracy[-1]:.4f}")
print(f"Training Precision: {train_precision[-1]:.4f}")
print(f"Validation Precision: {val_precision[-1]:.4f}")
print(f"Training Recall: {train_recall[-1]:.4f}")
print(f"Validation Recall: {val_recall[-1]:.4f}")

loss, accuracy, precision, recall = model.evaluate(test_padded, test_labels, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")


model.save('../models/regularized_rnn_model.keras')
clear_session()
