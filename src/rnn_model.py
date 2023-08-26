from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.regularizers import l1
from keras.metrics import Precision, Recall


def create_rnn_model(MAX_WORDS, MAX_LENGTH):
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LENGTH),
        LSTM(128, return_sequences=True),
        Dropout(0.6),
        LSTM(64),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model
