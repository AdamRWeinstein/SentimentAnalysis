from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.metrics import Precision, Recall


def create_rnn_model(MAX_WORDS, MAX_LENGTH):
    model = Sequential([
        Embedding(MAX_WORDS, 50, input_length=MAX_LENGTH),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model
