#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np
import pandas as pd
import keras
import tensorflow as tf

file_name = "all_data.csv"

df = pd.read_csv(file_name)
questions = df['question']
answers = df['answer']


# In[2]:


tokenizer = keras.preprocessing.text.Tokenizer()

def dataset_preparation(data):

    # basic cleanup
    corpus = data

    # tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # create input sequences using list of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # pad sequences 
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len, total_words

def create_model(predictors, label, max_sequence_len, total_words):

    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
    model.add(LSTM(150, return_sequences = True))
    # model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model.fit(predictors, label, epochs=100, verbose=1, callbacks=[earlystop])
    print( model.summary() )
    return model 

def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


# In[3]:


predictors, label, max_sequence_len, total_words = dataset_preparation(questions)
model = create_model(predictors, label, max_sequence_len, total_words)
# model.save_weights('model_weights.h5')
model.save('model.h5')


# In[4]:


output_sequences = []
for line in answers:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        output_sequences.append(n_gram_sequence)
# pad sequences 
max_output_sequences_len = max([len(x) for x in output_sequences])
# print(max_output_sequences_len)

# model = model.load_weights('model_weights.h5')
model = keras.models.load_model('model.h5')
print (generate_text("điểm chuẩn", max_output_sequences_len, max_sequence_len))


# In[ ]:




