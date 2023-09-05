from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from rnn_model import create_rnn_model

maxwords = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=maxwords)

maxlen = 500  # Maximum length of the sequences
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')


model_checkpoint = ModelCheckpoint('../models/model_LSTM_epoch_{epoch:02d}.keras',
                                   save_best_only=False,
                                   save_weights_only=False)
callbacks = [model_checkpoint]

# model = create_rnn_model(maxwords, maxlen)
model = load_model('../models/model_LSTM.keras')
history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), verbose=1, callbacks=callbacks)

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

model.save('../models/model_LSTM.keras')
