import keras.backend as k
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from keras.metrics import Recall, Precision
from keras.models import Sequential


# taken from old keras source code
def f1(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + k.epsilon())
    recall = true_positives / (possible_positives + k.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + k.epsilon())
    return f1_val


def custom_recall(y_true, y_pred):
    y_true = k.ones_like(y_true)
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))

    all_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    return true_positives / (all_positives + k.epsilon())


def custom_precision(y_true, y_pred):
    y_true = k.ones_like(y_true)
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))

    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + k.epsilon())


def f1_score(y_true, y_pred):
    precision = custom_precision(y_true, y_pred)
    recall = custom_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + k.epsilon()))


def create_lstm(x, y, model_name: str, num_words: int):
    metrics = ['accuracy', Precision(), Recall(), f1]

    model = Sequential(name=model_name)
    model.add(Embedding(num_words, 100, input_length=x.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
    return model
