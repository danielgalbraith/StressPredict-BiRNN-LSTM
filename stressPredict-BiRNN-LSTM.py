import pandas as pd
import numpy as np
from numpy import zeros
from pandas import DataFrame, get_dummies
import keras
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Bidirectional, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential, load_model
from keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping
from keras.constraints import max_norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import data from CSV
f = pd.read_csv('presidents-data-words-january-3-2018.csv')
df = DataFrame(f)
df = df.dropna(subset=['word','dagalb','alex'])

# Data preprocessing
dg_df = df[['word','dagalb']]
al_df = df[['word','alex']]
both_df = pd.concat([dg_df,al_df], ignore_index=True, sort=False)
both_df['annotations'] = both_df['dagalb'][:11639].append(both_df['alex'][11639:])

early_stop = EarlyStopping(patience=5)

def get_input(in_data, embed_size):
    """Takes pandas Series of type string as input, tokenizes and converts to padded sequences of the specified length parameter. Calls get_embs to get weights matrix from GloVe word vectors. Returns tuple of padded X matrix, embeddings matrix, vocab size and embedding size."""
    vocab_size = len(in_data)
    t = Tokenizer(num_words=vocab_size)
    t.fit_on_texts(in_data)
    seqs = t.texts_to_sequences(in_data)
    padded_X = pad_sequences(seqs, maxlen=embed_size)
    emb_matrix = get_embs(vocab_size, embed_size, t)
    return (padded_X, emb_matrix, vocab_size, embed_size)

def get_embs(vocab_size, embed_size, t):
    """Takes vocab size and embeddings size integers with tokenizer object as fitted to text, outputs matrix of word embeddings according to pre-trained GloVe 100D word vectors."""
    emb_idx = dict()
    f = open('glove.6B/glove.6B.100d.txt')
    for line in f:
        vals = line.split()
        word = vals[0]
        coefs = np.asarray(vals[1:], dtype='float32')
        emb_idx[word] = coefs
    f.close()
    emb_matrix = zeros((vocab_size, embed_size))
    for word, i in t.word_index.items():
        if i > vocab_size-1:
            break
        else:
            emb_vec = emb_idx.get(word)
        if emb_vec is not None:
            emb_matrix[i] = emb_vec
    return emb_matrix

# Prepare inputs
inputs = get_input(both_df['word'], 100)
padded_X = inputs[0]
emb_matrix = inputs[1]
vocab_size = inputs[2]
embed_size = inputs[3]

y = np.array(to_categorical(both_df.annotations))

# Split data into train and test sets
padded_X_train, padded_X_test, y_train, y_test = train_test_split(padded_X, y, test_size=0.2, random_state=42)

# Uncomment to load saved model:
#model=load_model('model.h5')

# Initiate model:
model = Sequential()
model.add(Embedding(vocab_size, embed_size, weights=[emb_matrix], input_length=100, trainable=False))
model.add(Dropout(0.2))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Bidirectional(LSTM(200, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(200, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(200, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(200, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Flatten())
model.add(Dense(7, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('model.h5')

# Fit and evaluate model
print(model.summary())
model.fit(padded_X_train, y_train, epochs=1, validation_split=0.2, batch_size=64, callbacks=[early_stop])
score = model.evaluate(padded_X_test, y_test, batch_size=64)
print("Accuracy on test set = " + format(score[1]*100, '.2f') + "%")

# Uncomment to display model architecture:
# plot_model(model, show_shapes=True)

# Test fitted model on unseen data
uf = open('unseen.txt','r')
unseen_text = uf.read()
uf.close()
unseen_df = DataFrame()
unseen_tokens_list = text_to_word_sequence(unseen_text)
unseen_tokens_series = pd.Series(unseen_tokens_list)
unseen_df['word'] = unseen_tokens_series.values
unseen_inputs = get_input(unseen_df['word'], 100)
unseen_X = unseen_inputs[0]
unseen_pred = model.predict_classes(unseen_X)
print(unseen_pred[0:10], unseen_df['word'][0:10])
