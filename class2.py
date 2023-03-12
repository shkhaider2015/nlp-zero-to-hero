# Sequencing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences)
# If you want to add zero after sentence
# If you dont want the padded sentence should be longer as the longest sentence
# you can specify desired length
# padded = pad_sequences(sequences, padding="post", maxlen=5)
# truncating="post"

print(f"Word Index :  {word_index}")
print(f"Sequences : {sequences}")
print(f"Pad Sequences : {padded}")
