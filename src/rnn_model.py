from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.regularizers import l2
from keras.metrics import Precision, Recall


def create_rnn_model(MAX_WORDS, MAX_LENGTH):
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LENGTH),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model


def create_rnn_model_stacked(MAX_WORDS, MAX_LENGTH):
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LENGTH),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model


def create_rnn_model_stacked_regularized(MAX_WORDS, MAX_LENGTH):
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LENGTH),
        LSTM(64, return_sequences=True,
             kernel_regularizer=l2(.001),
             recurrent_regularizer=l2(.001),
             bias_regularizer=l2(.001)),
        Dropout(0.2),
        LSTM(64,
             kernel_regularizer=l2(.001),
             recurrent_regularizer=l2(.001),
             bias_regularizer=l2(.001)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model
