from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from rnn_model import create_rnn_model
from sklearn.model_selection import KFold
import numpy as np

maxwords = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=maxwords)
# Combine the dataset since we're using KFold
x_data = np.concatenate((x_train, x_test), axis=0)
y_data = np.concatenate((y_train, y_test), axis=0)

maxlen = 500  # Maximum length of the sequences
x_data = pad_sequences(x_data, maxlen=maxlen, padding='post')

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True)

# Track the best model
best_model = None
best_val_loss = float('inf')

for train_index, val_index in kf.split(x_data):
    # Prepare the training and validation data for this fold
    x_train_fold, x_val_fold = x_data[train_index], x_data[val_index]
    y_train_fold, y_val_fold = y_data[train_index], y_data[val_index]

    model = create_rnn_model(maxwords, maxlen)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    callbacks = [early_stopping]

    history = model.fit(x_train_fold, y_train_fold, epochs=20, batch_size=64, validation_data=(x_val_fold, y_val_fold), verbose=1, callbacks=callbacks)

    final_val_loss = min(history.history['val_loss'])
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_model = model

    metric_map = {
        'accuracy': 'Train_Accuracy',
        'val_accuracy': 'Val_Accuracy',
        'precision': 'Train_Precision',
        'val_precision': 'Val_Precision',
        'recall': 'Train_Recall',
        'val_recall': 'Val_Recall',
        'loss': 'Train_Loss',
        'val_loss': 'Val_Loss'
    }

    for history_metric, df_metric in metric_map.items():
        print(f"\'{df_metric}\': [", end="")
        for i, value in enumerate(history.history[history_metric]):
            end = ", "
            if i == len(history.history[history_metric]) - 1:
                end = ""
            print(f"{value:.4f}", end=end)
        print("],")


if best_model is not None:
    best_model.save('../models/model_stacked_LSTM_50d_dofirst_folds.keras')
