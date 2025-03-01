import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, GRU, Dense, Dropout,
    SpatialDropout1D, Bidirectional, Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

TRAIN_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_training"
TEST_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing"
TEST_KEYS_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys"
GLOVE_PATH = "glove.6B/glove.6B.100d.txt"

MAX_VOCAB = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 20

from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        e = tf.keras.backend.squeeze(e, axis=-1)
        alpha = tf.keras.backend.softmax(e)
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context


def standardize_label(label):
    return re.sub(r"\(e1,e2\)|\(e2,e1\)", "", label).strip()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s<>/]", "", text)
    return text


def load_training_data(train_folder):
    sentences, labels = [], []
    train_file = "TRAIN_FILE.TXT"
    train_path = os.path.join(train_folder, train_file)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file {train_file} not found!")
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            sentence = lines[i].strip()
            label = standardize_label(lines[i + 1].strip())
            sentences.append(sentence)
            labels.append(label)
    print(f" Training Data Loaded: {len(sentences)} sentences, {len(labels)} labels")
    print(f" Unique Training Labels: {set(labels)}")
    return sentences, labels


train_sentences, train_labels = load_training_data(TRAIN_PATH)

train_sentences = [clean_text(sentence) for sentence in train_sentences]

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_onehot = to_categorical(train_labels_encoded)

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
word_index = tokenizer.word_index
train_padded_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")

X_train, X_val, y_train, y_val = train_test_split(train_padded_sequences, train_labels_onehot, test_size=0.2,
                                                  random_state=42)


def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


embedding_matrix = load_glove_embeddings(GLOVE_PATH, word_index, embedding_dim=EMBEDDING_DIM)


def build_enhanced_model():
    inputs = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=True)(inputs)
    x = SpatialDropout1D(0.2)(x)

    convs = []
    for kernel_size in [3, 4, 5]:
        conv = Conv1D(filters=256, kernel_size=kernel_size, activation="relu")(x)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    cnn_branch = Concatenate()(convs)

    gru_out = Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.2))(x)
    gru_out = Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.2))(gru_out)
    gru_pool = GlobalMaxPooling1D()(gru_out)
    gru_att = AttentionLayer()(gru_out)
    gru_branch = Concatenate()([gru_pool, gru_att])

    combined = Concatenate()([cnn_branch, gru_branch])
    combined = BatchNormalization()(combined)
    combined = Dense(128, activation="relu")(combined)
    combined = Dropout(0.5)(combined)
    outputs = Dense(len(label_encoder.classes_), activation="softmax")(combined)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


model = build_enhanced_model()
model.summary()

train_labels_int = label_encoder.transform(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels_int), y=train_labels_int)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(" Class Weights:", class_weights_dict)

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True,
                           verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)

history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    class_weight=class_weights_dict,
                    callbacks=[early_stop, reduce_lr])


def load_test_data(test_folder, test_labels_folder):
    test_sentences = {}
    test_labels = {}
    test_file = "TEST_FILE.TXT"
    test_path = os.path.join(test_folder, test_file)
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, sentence = parts
                test_sentences[sid] = clean_text(sentence)
    test_labels_file = "TEST_FILE_KEY.TXT"
    test_labels_path = os.path.join(test_labels_folder, test_labels_file)
    with open(test_labels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, label = parts
                test_labels[sid] = standardize_label(label)
    return test_sentences, test_labels


test_sentences_dict, test_labels_dict = load_test_data(TEST_PATH, TEST_KEYS_PATH)

print("\n🔍 Sample Test Sentences:", list(test_sentences_dict.items())[:5])
print("🔍 Unique Test Labels Before Encoding:", set(test_labels_dict.values()))

sorted_ids = sorted(test_sentences_dict.keys(), key=lambda x: int(x))
test_sentences_ordered = [test_sentences_dict[sid] for sid in sorted_ids]
test_labels_ordered = [test_labels_dict[sid] for sid in sorted_ids]

test_sequences = tokenizer.texts_to_sequences(test_sentences_ordered)
test_padded_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding="post")
test_labels_encoded = label_encoder.transform(test_labels_ordered)
test_labels_onehot = to_categorical(test_labels_encoded)

print(f" Test Label Shape: {test_labels_onehot.shape}, Expected: (num_samples, {len(label_encoder.classes_)})")

test_loss, test_acc = model.evaluate(test_padded_sequences, test_labels_onehot, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(test_padded_sequences)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(test_labels_onehot, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))


import os
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, GlobalMaxPooling1D, Dense, Dropout, \
    SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

TRAIN_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_training"
TEST_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing"
TEST_KEYS_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys"
GLOVE_PATH = "glove.6B/glove.6B.100d.txt"

MAX_VOCAB = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 20


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s<>/]", "", text)
    return text


def standardize_label(label):
    return re.sub(r"\(e1,e2\)|\(e2,e1\)", "", label).strip()


def load_training_data(folder):
    sentences, labels = [], []
    train_file = os.path.join(folder, "TRAIN_FILE.TXT")
    with open(train_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            sentence = lines[i].strip()
            label = standardize_label(lines[i + 1].strip())
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels


train_sentences, train_labels = load_training_data(TRAIN_PATH)
train_sentences = [clean_text(s) for s in train_sentences]

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_onehot = to_categorical(train_labels_encoded)

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
word_index = tokenizer.word_index
train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")

X_train, X_val, y_train, y_val = train_test_split(train_padded, train_labels_onehot, test_size=0.2, random_state=42)


def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


embedding_matrix = load_glove_embeddings(GLOVE_PATH, word_index, embedding_dim=EMBEDDING_DIM)


def build_rnn_model():
    inp = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.2))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(len(label_encoder.classes_), activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


model = build_rnn_model()
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)


def load_test_data(test_folder, test_labels_folder):
    test_sentences, test_labels = {}, {}
    test_file = os.path.join(test_folder, "TEST_FILE.TXT")
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, sentence = parts
                test_sentences[sid] = clean_text(sentence)
    test_labels_file = os.path.join(test_labels_folder, "TEST_FILE_KEY.TXT")
    with open(test_labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, label = parts
                test_labels[sid] = standardize_label(label)
    return test_sentences, test_labels


test_sentences_dict, test_labels_dict = load_test_data(TEST_PATH, TEST_KEYS_PATH)

sorted_ids = sorted(test_sentences_dict.keys(), key=lambda x: int(x))
test_sentences_ordered = [test_sentences_dict[sid] for sid in sorted_ids]
test_labels_ordered = [test_labels_dict[sid] for sid in sorted_ids]

test_sequences = tokenizer.texts_to_sequences(test_sentences_ordered)
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding="post")
test_labels_encoded = label_encoder.transform(test_labels_ordered)
test_labels_onehot = to_categorical(test_labels_encoded)

test_loss, test_acc = model.evaluate(test_padded, test_labels_onehot, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(test_padded)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(test_labels_onehot, axis=1)
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))


import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, GRU, Dense, Dropout,
    SpatialDropout1D, Bidirectional, Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import KeyedVectors

TRAIN_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_training"
TEST_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing"
TEST_KEYS_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys"
WORD2VEC_PATH = "GoogleNews-vectors-negative300.bin"

MAX_VOCAB = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 300
BATCH_SIZE = 32
EPOCHS = 20

from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        e = tf.keras.backend.squeeze(e, axis=-1)
        alpha = tf.keras.backend.softmax(e)
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context


def standardize_label(label):
    return re.sub(r"\(e1,e2\)|\(e2,e1\)", "", label).strip()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s<>/]", "", text)
    return text


def load_training_data(train_folder):
    sentences, labels = [], []
    train_file = "TRAIN_FILE.TXT"
    train_path = os.path.join(train_folder, train_file)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file {train_file} not found!")
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            sentence = lines[i].strip()
            label = standardize_label(lines[i + 1].strip())
            sentences.append(sentence)
            labels.append(label)
    print(f" Training Data Loaded: {len(sentences)} sentences, {len(labels)} labels")
    print(f" Unique Training Labels: {set(labels)}")
    return sentences, labels


train_sentences, train_labels = load_training_data(TRAIN_PATH)

train_sentences = [clean_text(sentence) for sentence in train_sentences]

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_onehot = to_categorical(train_labels_encoded)

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")

X_train, X_val, y_train, y_val = train_test_split(train_padded_sequences, train_labels_onehot, test_size=0.2,
                                                  random_state=42)


def load_word2vec_embeddings(word2vec_path, word_index, embedding_dim=300):
    print("Loading Word2Vec model...")
    w2v_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    print("Word2Vec model loaded.")

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in w2v_model:
            embedding_matrix[i] = w2v_model[word]
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix


embedding_matrix = load_word2vec_embeddings(WORD2VEC_PATH, word_index, EMBEDDING_DIM)


def build_enhanced_model():
    inputs = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=True)(inputs)
    x = SpatialDropout1D(0.2)(x)

    convs = []
    for kernel_size in [3, 4, 5]:
        conv = Conv1D(filters=256, kernel_size=kernel_size, activation="relu")(x)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    cnn_branch = Concatenate()(convs)

    gru_out = Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.2))(x)
    gru_out = Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.2))(gru_out)
    gru_pool = GlobalMaxPooling1D()(gru_out)
    gru_att = AttentionLayer()(gru_out)
    gru_branch = Concatenate()([gru_pool, gru_att])

    combined = Concatenate()([cnn_branch, gru_branch])
    combined = BatchNormalization()(combined)
    combined = Dense(128, activation="relu")(combined)
    combined = Dropout(0.5)(combined)
    outputs = Dense(len(label_encoder.classes_), activation="softmax")(combined)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


model = build_enhanced_model()
model.summary()

train_labels_int = label_encoder.transform(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels_int), y=train_labels_int)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(" Class Weights:", class_weights_dict)

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)

history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    class_weight=class_weights_dict,
                    callbacks=[early_stop, reduce_lr])


def load_test_data(test_folder, test_labels_folder):
    test_sentences = {}
    test_labels = {}
    test_file = "TEST_FILE.TXT"
    test_path = os.path.join(test_folder, test_file)
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, sentence = parts
                test_sentences[sid] = clean_text(sentence)
    test_labels_file = "TEST_FILE_KEY.TXT"
    test_labels_path = os.path.join(test_labels_folder, test_labels_file)
    with open(test_labels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, label = parts
                test_labels[sid] = standardize_label(label)
    return test_sentences, test_labels


test_sentences_dict, test_labels_dict = load_test_data(TEST_PATH, TEST_KEYS_PATH)

print("\n🔍 Sample Test Sentences:", list(test_sentences_dict.items())[:5])
print("🔍 Unique Test Labels Before Encoding:", set(test_labels_dict.values()))

sorted_ids = sorted(test_sentences_dict.keys(), key=lambda x: int(x))
test_sentences_ordered = [test_sentences_dict[sid] for sid in sorted_ids]
test_labels_ordered = [test_labels_dict[sid] for sid in sorted_ids]

test_sequences = tokenizer.texts_to_sequences(test_sentences_ordered)
test_padded_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding="post")
test_labels_encoded = label_encoder.transform(test_labels_ordered)
test_labels_onehot = to_categorical(test_labels_encoded)

print(f" Test Label Shape: {test_labels_onehot.shape}, Expected: (num_samples, {len(label_encoder.classes_)})")

test_loss, test_acc = model.evaluate(test_padded_sequences, test_labels_onehot, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(test_padded_sequences)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(test_labels_onehot, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))


import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, GRU, Dense, Dropout,
    SpatialDropout1D, Bidirectional, Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

TRAIN_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_training"
TEST_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing"
TEST_KEYS_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys"
FASTTEXT_PATH = "Unconfirmed 477500.crdownload/wiki.simple.bin"

MAX_VOCAB = 20000
MAX_LENGTH = 100
BATCH_SIZE = 32
EPOCHS = 10

from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context


def standardize_label(label):
    return re.sub(r"\(e1,e2\)|\(e2,e1\)", "", label).strip()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s<>/]", "", text)
    return text


def load_training_data(train_folder):
    sentences, labels = [], []
    train_file = "TRAIN_FILE.TXT"
    train_path = os.path.join(train_folder, train_file)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file {train_file} not found!")
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            sentence = lines[i].strip()
            label = standardize_label(lines[i + 1].strip())
            sentences.append(sentence)
            labels.append(label)
    print(f" Training Data Loaded: {len(sentences)} sentences, {len(labels)} labels")
    print(f" Unique Training Labels: {set(labels)}")
    return sentences, labels


train_sentences, train_labels = load_training_data(TRAIN_PATH)
train_sentences = [clean_text(sentence) for sentence in train_sentences]

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_onehot = to_categorical(train_labels_encoded)

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")

X_train, X_val, y_train, y_val = train_test_split(train_padded_sequences, train_labels_onehot, test_size=0.2,
                                                  random_state=42)

from gensim.models.fasttext import load_facebook_model


def load_fasttext_embeddings(fasttext_path, word_index, embedding_dim=300):
    print("Loading FastText model...")
    ft_model = load_facebook_model(fasttext_path)
    print("FastText model loaded.")

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in ft_model.wv:
            embedding_matrix[i] = ft_model.wv[word]
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix


embedding_matrix = load_fasttext_embeddings(FASTTEXT_PATH, word_index, EMBEDDING_DIM)


def build_enhanced_model():
    inputs = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=True)(inputs)
    x = SpatialDropout1D(0.2)(x)

    convs = []
    for kernel_size in [3, 4, 5]:
        conv = Conv1D(filters=256, kernel_size=kernel_size, activation="relu")(x)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    cnn_branch = Concatenate()(convs)

    gru_out = Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.2))(x)
    gru_out = Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.2))(gru_out)
    gru_pool = GlobalMaxPooling1D()(gru_out)
    gru_att = AttentionLayer()(gru_out)
    gru_branch = Concatenate()([gru_pool, gru_att])

    combined = Concatenate()([cnn_branch, gru_branch])
    combined = BatchNormalization()(combined)
    combined = Dense(128, activation="relu")(combined)
    combined = Dropout(0.5)(combined)
    outputs = Dense(len(label_encoder.classes_), activation="softmax")(combined)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


model = build_enhanced_model()
model.summary()

train_labels_int = label_encoder.transform(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels_int), y=train_labels_int)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(" Class Weights:", class_weights_dict)

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)

history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    class_weight=class_weights_dict,
                    callbacks=[early_stop, reduce_lr])


def load_test_data(test_folder, test_labels_folder):
    test_sentences = {}
    test_labels = {}
    test_file = "TEST_FILE.TXT"
    test_path = os.path.join(test_folder, test_file)
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, sentence = parts
                test_sentences[sid] = clean_text(sentence)
    test_labels_file = "TEST_FILE_KEY.TXT"
    test_labels_path = os.path.join(test_labels_folder, test_labels_file)
    with open(test_labels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, label = parts
                test_labels[sid] = standardize_label(label)
    return test_sentences, test_labels


test_sentences_dict, test_labels_dict = load_test_data(TEST_PATH, TEST_KEYS_PATH)
print("\n🔍 Sample Test Sentences:", list(test_sentences_dict.items())[:5])
print("🔍 Unique Test Labels Before Encoding:", set(test_labels_dict.values()))

sorted_ids = sorted(test_sentences_dict.keys(), key=lambda x: int(x))
test_sentences_ordered = [test_sentences_dict[sid] for sid in sorted_ids]
test_labels_ordered = [test_labels_dict[sid] for sid in sorted_ids]

test_sequences = tokenizer.texts_to_sequences(test_sentences_ordered)
test_padded_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding="post")
test_labels_encoded = label_encoder.transform(test_labels_ordered)
test_labels_onehot = to_categorical(test_labels_encoded)

print(f" Test Label Shape: {test_labels_onehot.shape}, Expected: (num_samples, {len(label_encoder.classes_)})")

test_loss, test_acc = model.evaluate(test_padded_sequences, test_labels_onehot, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(test_padded_sequences)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(test_labels_onehot, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))


import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, GRU, LSTM, Dense, Dropout,
    SpatialDropout1D, Bidirectional, Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

TRAIN_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_training"
TEST_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing"
TEST_KEYS_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys"
GLOVE_PATH = "glove.6B/glove.6B.100d.txt"

MAX_VOCAB = 20000
MAX_LENGTH = 100
BATCH_SIZE = 32


def standardize_label(label):
    return re.sub(r"\(e1,e2\)|\(e2,e1\)", "", label).strip()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s<>/]", "", text)
    return text


def load_training_data(train_folder):
    sentences, labels = [], []
    train_file = "TRAIN_FILE.TXT"
    train_path = os.path.join(train_folder, train_file)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file {train_file} not found!")
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            sentence = lines[i].strip()
            label = standardize_label(lines[i + 1].strip())
            sentences.append(sentence)
            labels.append(label)
    print(f" Training Data Loaded: {len(sentences)} sentences, {len(labels)} labels")
    print(f" Unique Training Labels: {set(labels)}")
    return sentences, labels


def load_test_data(test_folder, test_labels_folder):
    test_sentences = {}
    test_labels = {}
    test_file = "TEST_FILE.TXT"
    test_path = os.path.join(test_folder, test_file)
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, sentence = parts
                test_sentences[sid] = clean_text(sentence)
    test_labels_file = "TEST_FILE_KEY.TXT"
    test_labels_path = os.path.join(test_labels_folder, test_labels_file)
    with open(test_labels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, label = parts
                test_labels[sid] = standardize_label(label)
    return test_sentences, test_labels


def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


train_sentences, train_labels = load_training_data(TRAIN_PATH)
train_sentences = [clean_text(sentence) for sentence in train_sentences]

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_onehot = to_categorical(train_labels_encoded)
num_classes = len(label_encoder.classes_)

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")

X_train, X_val, y_train, y_val = train_test_split(train_padded_sequences, train_labels_onehot, test_size=0.2,
                                                  random_state=42)

embedding_matrix = load_glove_embeddings(GLOVE_PATH, word_index, EMBEDDING_DIM)

test_sentences_dict, test_labels_dict = load_test_data(TEST_PATH, TEST_KEYS_PATH)
sorted_ids = sorted(test_sentences_dict.keys(), key=lambda x: int(x))
test_sentences_ordered = [test_sentences_dict[sid] for sid in sorted_ids]
test_labels_ordered = [test_labels_dict[sid] for sid in sorted_ids]
test_sequences = tokenizer.texts_to_sequences(test_sentences_ordered)
test_padded_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding="post")
test_labels_encoded = label_encoder.transform(test_labels_ordered)
test_labels_onehot = to_categorical(test_labels_encoded)

print(f" Test Label Shape: {test_labels_onehot.shape}, Expected: (num_samples, {num_classes})")

train_labels_int = label_encoder.transform(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels_int), y=train_labels_int)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(" Class Weights:", class_weights_dict)



def build_cnn_model():
    inputs = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=True)(inputs)
    x = SpatialDropout1D(0.2)(x)
    convs = []
    for kernel_size in [3, 4, 5]:
        conv = Conv1D(filters=256, kernel_size=kernel_size, activation="relu")(x)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    x = Concatenate()(convs)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_gru_model():
    inputs = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=True)(inputs)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(128, recurrent_dropout=0.2, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_model():
    inputs = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=True)(inputs)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(128, recurrent_dropout=0.2, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)


def train_and_evaluate(model_builder, model_name):
    print("\n========== Training", model_name, "Model ==========")
    model = model_builder()
    model.summary()
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val),
                        class_weight=class_weights_dict,
                        callbacks=[early_stop, reduce_lr])

    test_loss, test_acc = model.evaluate(test_padded_sequences, test_labels_onehot, verbose=1)
    print(f"{model_name} Test Accuracy: {test_acc:.4f}")

    y_pred = model.predict(test_padded_sequences)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(test_labels_onehot, axis=1)

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))
    return model


model_cnn = train_and_evaluate(build_cnn_model, "CNN")

model_gru = train_and_evaluate(build_gru_model, "GRU")

model_lstm = train_and_evaluate(build_lstm_model, "LSTM")


import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, GRU, LSTM, Dense, Dropout,
    SpatialDropout1D, Bidirectional, Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

TRAIN_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_training"
TEST_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing"
TEST_KEYS_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys"

MAX_VOCAB = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 10


def standardize_label(label):
    return re.sub(r"\(e1,e2\)|\(e2,e1\)", "", label).strip()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s<>/]", "", text)
    return text


def load_training_data(train_folder):
    sentences, labels = [], []
    train_file = "TRAIN_FILE.TXT"
    train_path = os.path.join(train_folder, train_file)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file {train_file} not found!")
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            sentence = lines[i].strip()
            label = standardize_label(lines[i + 1].strip())
            sentences.append(sentence)
            labels.append(label)
    print(f" Training Data Loaded: {len(sentences)} sentences, {len(labels)} labels")
    print(f" Unique Training Labels: {set(labels)}")
    return sentences, labels


def load_test_data(test_folder, test_labels_folder):
    test_sentences = {}
    test_labels = {}
    test_file = "TEST_FILE.TXT"
    test_path = os.path.join(test_folder, test_file)
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, sentence = parts
                test_sentences[sid] = clean_text(sentence)
    test_labels_file = "TEST_FILE_KEY.TXT"
    test_labels_path = os.path.join(test_labels_folder, test_labels_file)
    with open(test_labels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, label = parts
                test_labels[sid] = standardize_label(label)
    return test_sentences, test_labels


train_sentences, train_labels = load_training_data(TRAIN_PATH)
train_sentences = [clean_text(sentence) for sentence in train_sentences]

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_onehot = to_categorical(train_labels_encoded)
num_classes = len(label_encoder.classes_)

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")

X_train, X_val, y_train, y_val = train_test_split(train_padded_sequences, train_labels_onehot, test_size=0.2,
                                                  random_state=42)

test_sentences_dict, test_labels_dict = load_test_data(TEST_PATH, TEST_KEYS_PATH)
sorted_ids = sorted(test_sentences_dict.keys(), key=lambda x: int(x))
test_sentences_ordered = [test_sentences_dict[sid] for sid in sorted_ids]
test_labels_ordered = [test_labels_dict[sid] for sid in sorted_ids]
test_sequences = tokenizer.texts_to_sequences(test_sentences_ordered)
test_padded_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding="post")
test_labels_encoded = label_encoder.transform(test_labels_ordered)
test_labels_onehot = to_categorical(test_labels_encoded)

print(f" Test Label Shape: {test_labels_onehot.shape}, Expected: (num_samples, {num_classes})")

train_labels_int = label_encoder.transform(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels_int), y=train_labels_int)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(" Class Weights:", class_weights_dict)



def build_cnn_model_random():
    inputs = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  input_length=MAX_LENGTH,
                  trainable=True)(inputs)
    x = SpatialDropout1D(0.2)(x)
    convs = []
    for kernel_size in [3, 4, 5]:
        conv = Conv1D(filters=256, kernel_size=kernel_size, activation="relu")(x)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    x = Concatenate()(convs)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_gru_model_random():
    inputs = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  input_length=MAX_LENGTH,
                  trainable=True)(inputs)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(128, recurrent_dropout=0.2, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_model_random():
    inputs = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  input_length=MAX_LENGTH,
                  trainable=True)(inputs)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(128, recurrent_dropout=0.2, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)


def train_and_evaluate(model_builder, model_name):
    print("\n========== Training", model_name, "Model ==========")
    model = model_builder()
    model.summary()
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val),
                        class_weight=class_weights_dict,
                        callbacks=[early_stop, reduce_lr])

    test_loss, test_acc = model.evaluate(test_padded_sequences, test_labels_onehot, verbose=1)
    print(f"{model_name} Test Accuracy: {test_acc:.4f}")

    y_pred = model.predict(test_padded_sequences)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(test_labels_onehot, axis=1)

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))
    return model



model_cnn_random = train_and_evaluate(build_cnn_model_random, "CNN (Random Embeddings)")

model_gru_random = train_and_evaluate(build_gru_model_random, "GRU (Random Embeddings)")

model_lstm_random = train_and_evaluate(build_lstm_model_random, "LSTM (Random Embeddings)")


import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, SpatialDropout1D, \
    Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

TRAIN_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_training"
TEST_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing"
TEST_KEYS_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys"
GLOVE_PATH = "glove.6B/glove.6B.100d.txt"

MAX_VOCAB = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32


def standardize_label(label):
    return re.sub(r"\(e1,e2\)|\(e2,e1\)", "", label).strip()


def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9\s<>/]", "", text)


def load_training_data(train_folder):
    sentences, labels = [], []
    train_file = "TRAIN_FILE.TXT"
    train_path = os.path.join(train_folder, train_file)
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            sentences.append(lines[i].strip())
            labels.append(standardize_label(lines[i + 1].strip()))
    print(f" Training Data Loaded: {len(sentences)} sentences, {len(labels)} labels")
    print(f" Unique Training Labels: {set(labels)}")
    return sentences, labels


def load_test_data(test_folder, test_labels_folder):
    test_sentences = {}
    test_labels = {}
    test_file = "TEST_FILE.TXT"
    with open(os.path.join(test_folder, test_file), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, sentence = parts
                test_sentences[sid] = clean_text(sentence)
    test_labels_file = "TEST_FILE_KEY.TXT"
    with open(os.path.join(test_labels_folder, test_labels_file), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, label = parts
                test_labels[sid] = standardize_label(label)
    return test_sentences, test_labels


def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
    return embedding_matrix


train_sentences, train_labels = load_training_data(TRAIN_PATH)
train_sentences = [clean_text(s) for s in train_sentences]
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_onehot = to_categorical(train_labels_encoded)
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")
X_train, X_val, y_train, y_val = train_test_split(train_padded, train_labels_onehot, test_size=0.2, random_state=42)
embedding_matrix = load_glove_embeddings(GLOVE_PATH, word_index, EMBEDDING_DIM)
test_sentences_dict, test_labels_dict = load_test_data(TEST_PATH, TEST_KEYS_PATH)
sorted_ids = sorted(test_sentences_dict.keys(), key=lambda x: int(x))
test_sentences = [test_sentences_dict[sid] for sid in sorted_ids]
test_labels = [test_labels_dict[sid] for sid in sorted_ids]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding="post")
test_labels_encoded = label_encoder.transform(test_labels)
test_onehot = to_categorical(test_labels_encoded)
num_classes = len(label_encoder.classes_)
train_labels_int = label_encoder.transform(train_labels)
cw = compute_class_weight("balanced", classes=np.unique(train_labels_int), y=train_labels_int)
class_weights = {i: weight for i, weight in enumerate(cw)}


def build_cnn_finetuned():
    inp = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)
    convs = []
    for k in [3, 4, 5]:
        conv = Conv1D(filters=256, kernel_size=k, activation="relu")(x)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    x = Concatenate()(convs)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
]

model_finetuned = build_cnn_finetuned()
model_finetuned.summary()
model_finetuned.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val), class_weight=class_weights,
                    callbacks=callbacks)
loss_ft, acc_ft = model_finetuned.evaluate(test_padded, test_onehot, verbose=1)
print(f"Fine-Tuned Embeddings CNN Test Accuracy: {acc_ft:.4f}")
print(classification_report(np.argmax(test_onehot, axis=1),
                            np.argmax(model_finetuned.predict(test_padded), axis=1),
                            target_names=label_encoder.classes_))


import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, GRU, LSTM, Dense, Dropout, \
    SpatialDropout1D, Bidirectional, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

TRAIN_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_training"
TEST_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing"
TEST_KEYS_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys"
GLOVE_PATH = "glove.6B/glove.6B.100d.txt"

MAX_VOCAB = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 10


def standardize_label(label):
    return re.sub(r"\(e1,e2\)|\(e2,e1\)", "", label).strip()


def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9\s<>/]", "", text)


def load_training_data(train_folder):
    sentences, labels = [], []
    train_file = "TRAIN_FILE.TXT"
    with open(os.path.join(train_folder, train_file), "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            sentences.append(lines[i].strip())
            labels.append(standardize_label(lines[i + 1].strip()))
    print(f"Loaded {len(sentences)} training sentences.")
    return sentences, labels


def load_test_data(test_folder, test_labels_folder):
    test_sentences = {}
    test_labels = {}
    test_file = "TEST_FILE.TXT"
    with open(os.path.join(test_folder, test_file), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                test_sentences[parts[0]] = clean_text(parts[1])
    test_labels_file = "TEST_FILE_KEY.TXT"
    with open(os.path.join(test_labels_folder, test_labels_file), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                test_labels[parts[0]] = standardize_label(parts[1])
    return test_sentences, test_labels


def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
    return embedding_matrix


train_sentences, train_labels = load_training_data(TRAIN_PATH)
train_sentences = [clean_text(s) for s in train_sentences]
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_onehot = to_categorical(train_labels_encoded)
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")
X_train, X_val, y_train, y_val = train_test_split(train_padded, train_labels_onehot, test_size=0.2, random_state=42)
embedding_matrix = load_glove_embeddings(GLOVE_PATH, word_index, EMBEDDING_DIM)
test_sentences_dict, test_labels_dict = load_test_data(TEST_PATH, TEST_KEYS_PATH)
sorted_ids = sorted(test_sentences_dict.keys(), key=lambda x: int(x))
test_sentences = [test_sentences_dict[sid] for sid in sorted_ids]
test_labels = [test_labels_dict[sid] for sid in sorted_ids]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding="post")
test_labels_encoded = label_encoder.transform(test_labels)
test_onehot = to_categorical(test_labels_encoded)
num_classes = len(label_encoder.classes_)
train_labels_int = label_encoder.transform(train_labels)
cw = compute_class_weight("balanced", classes=np.unique(train_labels_int), y=train_labels_int)
class_weights = {i: weight for i, weight in enumerate(cw)}


def build_cnn_model():
    inp = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)
    convs = []
    for k in [3, 4, 5]:
        conv = Conv1D(filters=256, kernel_size=k, activation="relu")(x)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    x = Concatenate()(convs)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_gru_model():
    inp = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(128, recurrent_dropout=0.2, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_model():
    inp = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(128, recurrent_dropout=0.2, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


callbacks = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
             ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)]

print("Training CNN model...")
model_cnn = build_cnn_model()
model_cnn.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val), class_weight=class_weights, callbacks=callbacks)

print("Training GRU model...")
model_gru = build_gru_model()
model_gru.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val), class_weight=class_weights, callbacks=callbacks)

print("Training LSTM model...")
model_lstm = build_lstm_model()
model_lstm.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
               validation_data=(X_val, y_val), class_weight=class_weights, callbacks=callbacks)

pred_cnn = model_cnn.predict(test_padded)
pred_gru = model_gru.predict(test_padded)
pred_lstm = model_lstm.predict(test_padded)
ensemble_pred = (pred_cnn + pred_gru + pred_lstm) / 3.0
ensemble_labels = np.argmax(ensemble_pred, axis=1)
true_labels = np.argmax(test_onehot, axis=1)
print("\nEnsemble Classification Report:")
print(classification_report(true_labels, ensemble_labels, target_names=label_encoder.classes_))


import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, SpatialDropout1D, \
    Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

TRAIN_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_training"
TEST_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing"
TEST_KEYS_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys"
GLOVE_PATH = "glove.6B/glove.6B.100d.txt"

MAX_VOCAB = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 20


def standardize_label(label):
    return re.sub(r"\(e1,e2\)|\(e2,e1\)", "", label).strip()


def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9\s<>/]", "", text)


def load_training_data(train_folder):
    sentences, labels = [], []
    train_file = "TRAIN_FILE.TXT"
    train_path = os.path.join(train_folder, train_file)
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            sentences.append(lines[i].strip())
            labels.append(standardize_label(lines[i + 1].strip()))
    print(f" Training Data Loaded: {len(sentences)} sentences, {len(labels)} labels")
    print(f" Unique Training Labels: {set(labels)}")
    return sentences, labels


def load_test_data(test_folder, test_labels_folder):
    test_sentences = {}
    test_labels = {}
    test_file = "TEST_FILE.TXT"
    with open(os.path.join(test_folder, test_file), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, sentence = parts
                test_sentences[sid] = clean_text(sentence)
    test_labels_file = "TEST_FILE_KEY.TXT"
    with open(os.path.join(test_labels_folder, test_labels_file), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, label = parts
                test_labels[sid] = standardize_label(label)
    return test_sentences, test_labels


def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
    return embedding_matrix


train_sentences, train_labels = load_training_data(TRAIN_PATH)
train_sentences = [clean_text(s) for s in train_sentences]
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_onehot = to_categorical(train_labels_encoded)
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")
X_train, X_val, y_train, y_val = train_test_split(train_padded, train_labels_onehot, test_size=0.2, random_state=42)
embedding_matrix = load_glove_embeddings(GLOVE_PATH, word_index, EMBEDDING_DIM)
test_sentences_dict, test_labels_dict = load_test_data(TEST_PATH, TEST_KEYS_PATH)
sorted_ids = sorted(test_sentences_dict.keys(), key=lambda x: int(x))
test_sentences = [test_sentences_dict[sid] for sid in sorted_ids]
test_labels = [test_labels_dict[sid] for sid in sorted_ids]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding="post")
test_labels_encoded = label_encoder.transform(test_labels)
test_onehot = to_categorical(test_labels_encoded)
num_classes = len(label_encoder.classes_)
train_labels_int = label_encoder.transform(train_labels)
cw = compute_class_weight("balanced", classes=np.unique(train_labels_int), y=train_labels_int)
class_weights = {i: weight for i, weight in enumerate(cw)}


def build_cnn_frozen():
    inp = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    convs = []
    for k in [3, 4, 5]:
        conv = Conv1D(filters=256, kernel_size=k, activation="relu")(x)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    x = Concatenate()(convs)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
]

model_frozen = build_cnn_frozen()
model_frozen.summary()

model_frozen.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                 validation_data=(X_val, y_val), class_weight=class_weights,
                 callbacks=callbacks)

loss_frozen, acc_frozen = model_frozen.evaluate(test_padded, test_onehot, verbose=1)
print(f"Frozen Embeddings CNN Test Accuracy: {acc_frozen:.4f}")
print(classification_report(np.argmax(test_onehot, axis=1),
                            np.argmax(model_frozen.predict(test_padded), axis=1),
                            target_names=label_encoder.classes_))


import os
import re
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, GlobalMaxPooling1D,
                                     Bidirectional, GRU, Dense, Dropout,
                                     SpatialDropout1D, Concatenate, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

TRAIN_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_training"
TEST_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing"
TEST_KEYS_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys"
GLOVE_PATH = "glove.6B/glove.6B.100d.txt"

MAX_VOCAB = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 20


def standardize_label(label):
    return re.sub(r"\(e1,e2\)|\(e2,e1\)", "", label).strip()


def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9\s<>/]", "", text)


def load_training_data(folder):
    sentences, labels = [], []
    train_file = os.path.join(folder, "TRAIN_FILE.TXT")
    with open(train_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            sentence = lines[i].strip()
            label = standardize_label(lines[i + 1].strip())
            sentences.append(sentence)
            labels.append(label)
    print(f" Training Data Loaded: {len(sentences)} sentences, {len(labels)} labels")
    print(f" Unique Training Labels: {set(labels)}")
    return sentences, labels


train_sentences, train_labels = load_training_data(TRAIN_PATH)
train_sentences = [clean_text(s) for s in train_sentences]

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_onehot = to_categorical(train_labels_encoded)

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
word_index = tokenizer.word_index
train_padded_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")

X_train, X_val, y_train, y_val = train_test_split(train_padded_sequences, train_labels_onehot, test_size=0.2,
                                                  random_state=42)


def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


embedding_matrix = load_glove_embeddings(GLOVE_PATH, word_index, EMBEDDING_DIM)

from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        e = tf.keras.backend.squeeze(e, axis=-1)
        alpha = tf.keras.backend.softmax(e)
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context


def build_hybrid_model():
    inputs = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=len(word_index) + 1,
                  output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=True)(inputs)
    x = SpatialDropout1D(0.2)(x)

    conv1 = Conv1D(filters=128, kernel_size=3, activation="relu")(x)
    pool1 = GlobalMaxPooling1D()(conv1)

    rnn_out = Bidirectional(GRU(64, return_sequences=True, recurrent_dropout=0.2))(x)
    rnn_out = Bidirectional(GRU(64, return_sequences=True, recurrent_dropout=0.2))(rnn_out)
    pool2 = GlobalMaxPooling1D()(rnn_out)
    attn = AttentionLayer()(rnn_out)

    rnn_combined = Concatenate()([pool2, attn])

    combined = Concatenate()([pool1, rnn_combined])
    combined = BatchNormalization()(combined)
    combined = Dense(64, activation="relu")(combined)
    combined = Dropout(0.5)(combined)
    outputs = Dense(len(label_encoder.classes_), activation="softmax")(combined)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


model = build_hybrid_model()
model.summary()

train_labels_int = label_encoder.transform(train_labels)
cw = compute_class_weight("balanced", classes=np.unique(train_labels_int), y=train_labels_int)
class_weights_dict = {i: weight for i, weight in enumerate(cw)}
print(" Class Weights:", class_weights_dict)

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)

history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    class_weight=class_weights_dict,
                    callbacks=[early_stop, reduce_lr])


def load_test_data(test_folder, test_labels_folder):
    test_sentences = {}
    test_labels = {}
    test_file = os.path.join(test_folder, "TEST_FILE.TXT")
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, sentence = parts
                test_sentences[sid] = clean_text(sentence)
    test_labels_file = os.path.join(test_labels_folder, "TEST_FILE_KEY.TXT")
    with open(test_labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, label = parts
                test_labels[sid] = standardize_label(label)
    return test_sentences, test_labels


test_sentences_dict, test_labels_dict = load_test_data(TEST_PATH, TEST_KEYS_PATH)
print("\n🔍 Sample Test Sentences:", list(test_sentences_dict.items())[:5])
print("🔍 Unique Test Labels:", set(test_labels_dict.values()))

sorted_ids = sorted(test_sentences_dict.keys(), key=lambda x: int(x))
test_sentences_ordered = [test_sentences_dict[sid] for sid in sorted_ids]
test_labels_ordered = [test_labels_dict[sid] for sid in sorted_ids]

test_sequences = tokenizer.texts_to_sequences(test_sentences_ordered)
test_padded_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding="post")
test_labels_encoded = label_encoder.transform(test_labels_ordered)
test_labels_onehot = to_categorical(test_labels_encoded)

print(f" Test Label Shape: {test_labels_onehot.shape}, Expected: (num_samples, {len(label_encoder.classes_)})")

test_loss, test_acc = model.evaluate(test_padded_sequences, test_labels_onehot, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(test_padded_sequences)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(test_labels_onehot, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))

import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, GlobalMaxPooling1D, Dropout, BatchNormalization,
    Concatenate, Bidirectional, GRU, SpatialDropout1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

TRAIN_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_training"
TEST_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing"
TEST_KEYS_PATH = "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys"
BERT_MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 100
BATCH_SIZE = 16
EPOCHS = 10
EMBEDDING_DIM = 768

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


def clean_text(text):
    return text.lower()


def load_training_data(train_folder):
    sentences, labels = [], []
    train_file = os.path.join(train_folder, "TRAIN_FILE.TXT")
    with open(train_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            sentence = lines[i].strip()
            label = lines[i + 1].strip()
            sentences.append(clean_text(sentence))
            labels.append(label)
    return sentences, labels


train_sentences, train_labels = load_training_data(TRAIN_PATH)

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
train_labels_onehot = to_categorical(train_labels_encoded)


def encode_texts(sentences):
    return tokenizer(sentences, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors="tf")


train_encodings = encode_texts(train_sentences)

X_train, X_val, y_train, y_val = train_test_split(
    train_encodings["input_ids"], train_labels_onehot, test_size=0.2, random_state=42
)


def build_hybrid_model():
    bert_input = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_word_ids")

    bert_model = TFBertModel.from_pretrained(BERT_MODEL_NAME)
    bert_output = bert_model(bert_input)[0]

    x = SpatialDropout1D(0.2)(bert_output)

    conv1 = Conv1D(filters=256, kernel_size=3, activation="relu")(x)
    cnn_pool = GlobalMaxPooling1D()(conv1)

    rnn = Bidirectional(GRU(128, return_sequences=True))(x)
    rnn_pool = GlobalMaxPooling1D()(rnn)

    combined = Concatenate()([cnn_pool, rnn_pool])
    combined = BatchNormalization()(combined)
    combined = Dense(128, activation="relu")(combined)
    combined = Dropout(0.5)(combined)

    outputs = Dense(len(label_encoder.classes_), activation="softmax")(combined)

    model = Model(inputs=bert_input, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


model = build_hybrid_model()
model.summary()

train_labels_int = label_encoder.transform(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels_int), y=train_labels_int)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(" Class Weights:", class_weights_dict)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    class_weight=class_weights_dict
)


def load_test_data(test_folder, test_labels_folder):
    test_sentences, test_labels = {}, {}
    test_file = os.path.join(test_folder, "TEST_FILE.TXT")
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, sentence = parts
                test_sentences[sid] = clean_text(sentence)

    test_labels_file = os.path.join(test_labels_folder, "TEST_FILE_KEY.TXT")
    with open(test_labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sid, label = parts
                test_labels[sid] = label

    return test_sentences, test_labels


test_sentences_dict, test_labels_dict = load_test_data(TEST_PATH, TEST_KEYS_PATH)

sorted_ids = sorted(test_sentences_dict.keys(), key=lambda x: int(x))
test_sentences_ordered = [test_sentences_dict[sid] for sid in sorted_ids]
test_labels_ordered = [test_labels_dict[sid] for sid in sorted_ids]

test_encodings = encode_texts(test_sentences_ordered)
test_labels_encoded = label_encoder.transform(test_labels_ordered)
test_labels_onehot = to_categorical(test_labels_encoded)

test_loss, test_acc = model.evaluate(test_encodings["input_ids"], test_labels_onehot, verbose=1)
print(f" Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(test_encodings["input_ids"])
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(test_labels_onehot, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))
