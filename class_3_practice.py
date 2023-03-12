import tensorflow as tf

new_model = tf.keras.models.load_model('nlp_class_3_modal.h5')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Show the model architecture
new_model.summary()

# predict
max_length = 100
padding_type = 'post'
trunc_type = 'post'
vocab_size = 10000

sentence = [
    "granny starting to fear spiders in the garden might be real",
    "game of thrones season finale showing this sunday night"
]
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(sentence)
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=100, padding="post", truncating="post")

print(new_model.predict(padded))