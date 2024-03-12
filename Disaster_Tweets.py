#Importing all libraries

import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import regex as re
import sklearn
from sklearn.model_selection import train_test_split
import keras.backend as K



#Train Section

#cleaning the data - remove links,lowercase,remove special characters

df = pd.read_csv("train.csv")

def clean_text(df):
    df["text"] = [re.sub(r'http\S+','',x,flags=re.MULTILINE) for x in df["text"]]
    df["text"] = df["text"].str.lower()
    translator = str.maketrans('', '', "!@#$%^&*()_+{}[];:,<>.?/\\|`~")
    df["text"] = [text.translate(translator) for text in df["text"]]
clean_text(df)

#Splitting data into training and test set - approx 85% 15% split

sentences = [x for x in df["text"]]
labels = [y for y in df["target"]]
labels = np.array(labels)
print(len(sentences))


training_sentences = sentences[:6090]
training_labels = labels[:6090]
testing_sentences = sentences[6090:]
testing_labels = labels[6090:]




#Setting all the parameters that need to be find into the model

vocab_size = 10000
embedding_dim = 16
max_length = 280
trunc_type='post'
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequence = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded_sequence = pad_sequences(testing_sequences,maxlen=max_length)


#Bulding the model 

my_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

my_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

(my_model.summary())

np.random.seed(42)

num_epochs = 10

my_model.fit(padded_sequence,training_labels,epochs=num_epochs,
validation_data = (testing_padded_sequence,testing_labels))

#Test Section



# Load the test data
test_df = pd.read_csv("test.csv")

# Clean and preprocess the test data
clean_text(test_df)
test_sentences = [x for x in test_df["text"]]

# Tokenize and pad the test data
test_sequences = tokenizer.texts_to_sequences(test_sentences)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length)

# Make predictions on the test data
predictions = my_model.predict(padded_test_sequences)

threshold = 0.5  # You can adjust the threshold as needed
predicted_labels = (predictions > threshold).astype(int)

# Match predictions with corresponding IDs
test_ids = test_df["id"]

# Create a DataFrame containing IDs and predictions
submission_df = pd.DataFrame({"id": test_ids, "target": predicted_labels.flatten()})

# Save the DataFrame to a CSV file for submission
submission_df.to_csv("submission.csv", index=False)

