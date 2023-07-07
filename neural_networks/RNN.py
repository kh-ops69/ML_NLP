import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.corpus import words, stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.compose import ColumnTransformer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn import svm
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Dense, Input, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.python.keras.layers import LSTM, GRU, SimpleRNN, Embedding
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import adamax_v2
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
pd.set_option("display.max_colwidth", None)

# labeled_df = pd.read_csv("uci-news-aggregator.csv")
# # print(labeled_df.sample(5))
# # print(labeled_df.shape, labeled_df.keys())
# # print(labeled_df['neither'].value_counts())
# # print(np.where(labeled_df['neither']==0))
# # print(labeled_df['neither'][1])
# labeled_df = labeled_df.drop(['URL', 'ID', 'PUBLISHER', 'STORY', 'HOSTNAME', 'TIMESTAMP'], axis=1)
# neutral_df = labeled_df.groupby('CATEGORY', group_keys=False).apply(lambda x: x['TITLE'].sample(3000))
# neutral_df = pd.DataFrame(neutral_df, columns=['TITLE'])
# neutral_df.to_csv('/Users/krishhashia/PycharmProjects/GRAPHS/data/neutral_dataset1', columns=['TITLE'], index=None)
# print(neutral_df.columns, neutral_df.TITLE.sample(5))

df = pd.read_csv('data/original/TwitterDF.csv')
df = df.drop(['Unnamed: 0'], axis=1)
print(df.columns)

df_train, df_test = train_test_split(df, test_size=0.4, random_state=13243)
max_vocab_size = 10000
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(df_train.TITLE)
train_sequences = tokenizer.texts_to_sequences(df_train.TITLE)
test_sequences = tokenizer.texts_to_sequences(df_test.TITLE)

word_idx = tokenizer.word_index
V = len(word_idx)
print("number of unique tokens: %d",len(word_idx))

train_data = pad_sequences(train_sequences)
T = train_data.shape[1]
print(f"shape of train data: {T}")

test_data = pad_sequences(test_sequences, maxlen=T)
print(f"shape of test data: {test_data.shape}")

# setting hyperparameter d to 30 (how much information model can store with respect to each token
d = 30
K = 2
i = Input(shape=(T,))
j = Embedding(V+1, d)(i)
j = LSTM(50, return_sequences=True)(j)
j = GlobalAveragePooling1D()(j)
j = Dense(K)(j)
# setting dense layer K=2 since it is binary classification: we'll have two outputs so this dense layer is 2D

model = Model(i, j)
# model's shape is  initialized with the input and dense layer outputs

model.compile(loss='binary_crossentropy', optimizer=adamax_v2.Adamax(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00002),
              metrics=['accuracy'])
print('training model...')
r = model.fit(train_data, df_train.label, epochs=25, validation_data=(test_data, df_test['label']))

different_test = pd.read_csv('data/original/test.csv')
different_test_sequences = tokenizer.texts_to_sequences(different_test.tweet)
different_test_data = pad_sequences(different_test_sequences, maxlen=T)
predictions = model.predict(different_test_data)
predicted_labels = np.argmax(predictions, axis=1)
different_test['predictions'] = predicted_labels
print(different_test.sample(10))

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

