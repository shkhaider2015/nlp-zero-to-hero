import json
import sys

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

with open("Dataset/Sarcasm_Headlines_Dataset.json", 'r') as f:
    datastore = []
    for line in f:
        line_data = json.loads(line)
        datastore.append(line_data)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])


training_size = 20000
vocab_size = 10000
padding_type = "post"
trunc_type = "post"
max_length = 100
embadding_dim = 16

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequencs = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequencs, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequencs = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequencs, maxlen=max_length, padding=padding_type, truncating=trunc_type)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embadding_dim, input_length=max_length ),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 30

history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)




# def plot_graphs(history, string):
#     plt.plot(history.history[string])
#     plt.plot(history.history['val_' + string])
#     plt.xlabel("Epochs")
#     plt.ylabel(string)
#     plt.legend([string, 'val_' + string])
#     plt.show()
#
#
# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")


# Save modal
model.save("nlp_class_3_modal.h5")
