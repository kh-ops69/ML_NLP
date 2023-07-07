import random

import gensim.downloader as api
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, Input, Embedding, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from sklearn.neighbors import NearestNeighbors
from keras_preprocessing.text import Tokenizer

dataset = api.load('text8')
# print(type(dataset))

# i = 0
# for x in dataset:
#   print(x)
#   i += 1
#   if i > 10:
#     break

# print(api.info())
# length=0
# record_lengths = []
# for record in dataset:
#   length += 1
#   record_lengths.append(len(record))
# print(length)
# plt.hist(record_lengths, bins=100)
# setting bins param to divide dataset into 100 parts and check for occurrences in each part
# plt.show()
# print(np.mean(record_lengths), np.std(record_lengths))

vocab_size = 20000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(dataset)
dataset_sequences = tokenizer.texts_to_sequences(dataset)


context_size = 20
embedding_dims = 100

i = Input(shape=(context_size,))
x = Embedding(vocab_size, embedding_dims)(i)
x = Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)
x = Dense(vocab_size, use_bias=False)(x)

model = Model(i, x)
print(model.summary())


def custom_generator(data, batch_size=256):
    train_matrix = np.zeros((batch_size, context_size))
    target_matrix = np.zeros(batch_size)
    required_iters = int(np.ceil(len(data) / batch_size))
    while True:
        random.shuffle(data)
        for j in range(required_iters):
            batch_sequences = data[j * batch_size:(j + 1) * batch_size]
            current_batch_size = len(batch_sequences)
            for k in range(current_batch_size):
                words_vector = batch_sequences[k]
                a = np.random.randint(0, len(words_vector) - context_size - 1)
                # last word in the document or sentence needs some context words ahead of it so we subtract it
                # initially only choosing word index from sequence, which will be taken as context window
                left_words = words_vector[a:a + (context_size // 2)]
                right_words = words_vector[a + (context_size // 2) + 1:a + context_size + 1]
                # we cannot take index as a-(context_size//2) first because if a is 0 it will return error
                total_words = left_words+right_words
                train_matrix[k] = total_words
                target_vals = words_vector[a+(context_size//2)]
                target_matrix[k] = target_vals
                yield train_matrix[:current_batch_size], target_matrix[:current_batch_size]
                # because last batch might be of different size, this way allows us to handle variable sized batches


model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics='accuracy')
Batch_size = 128
r = model.fit(custom_generator(dataset_sequences, Batch_size), epochs=5000, steps_per_epoch=int(np.ceil(len(dataset_sequences)//Batch_size)))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['accuracy'], label='accuracy')
plt.legend()
plt.show()

embeddings = model.layers[1].get_weights()[0]     # getting weights associated with each word
neighbours = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
neighbours.fit(embeddings)

def print_similar_words(query):
    query_idx = tokenizer.word_index[query]
    embedding_vect = embeddings[query_idx:query_idx+1]
    distances, indices = neighbours.kneighbors(embedding_vect)
    for i in indices[0]:
        # indices of words in first row from the vector obtained from kneighbours
        word = tokenizer.index_word[i]
        print(word)

print_similar_words('racist')
print_similar_words('titanic')
print_similar_words('child')