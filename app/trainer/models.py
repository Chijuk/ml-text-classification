import tensorflow.keras.backend as k
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Conv1D
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.models import Sequential


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
    model.add(Embedding(input_dim=num_words, output_dim=100, input_length=x.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
    return model


def create_embed_model(x, y, model_name: str, num_words: int):
    metrics = ['accuracy', Precision(), Recall(), f1]

    model = Sequential(name=model_name)
    model.add(Embedding(input_dim=num_words, output_dim=100, input_length=x.shape[1]))
    model.add(GlobalMaxPool1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
    return model


# CNNs work best with large training sets where they are able to find generalizations where a simple model like
# logistic regression won’t be able.
def create_cnn(x, y, model_name: str, num_words: int):
    metrics = ['accuracy', Precision(), Recall(), f1]

    model = Sequential(name=model_name)
    model.add(Embedding(input_dim=num_words, output_dim=100, input_length=x.shape[1]))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
    return model
