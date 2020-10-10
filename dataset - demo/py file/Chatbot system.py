#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras import layers , activations , models , preprocessing , utils
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense, Dropout, Input
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np
import pandas as pd
import keras

file_name = "all_data.csv"

df = pd.read_csv(file_name)
questions = df['question']
# print(questions[:5])
answers = df['answer']
# print(answers[:5])
labels = df['label']
# print(labels[:5])


# In[ ]:


import re 
from nltk.tokenize import word_tokenize
def replace_words(text): 
#     LatinChar = '[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]' 
    SpecialAndSpaceChar = '[/^$*+?#!@{}&\n\t\f\r]'
    #Xử lý lọc dữ liệu
#     deleteLatin = re.sub(LatinChar, '', text).strip()
    text = re.sub(SpecialAndSpaceChar, '', text).strip()
    return text
questions = questions.apply(replace_words)
# answers = answers.apply(replace_words)
# print(questions[100])


# In[ ]:


tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( questions ) 
tokenized_question = tokenizer.texts_to_sequences( questions ) 
# print(tokenized_question[:5])

length_list = list()
for token_seq in questions:
    length_list.append( len( token_seq ))
max_input_length = np.array( length_list ).max()

padded_question = keras.preprocessing.sequence.pad_sequences( tokenized_question , maxlen=max_input_length , padding='post' )
encoder_input_data = np.array( padded_question )

question_word_dict = tokenizer.word_index
num_question_tokens = len( question_word_dict )+1

mar_answer = list()
for line in answers:
    mar_answer.append( '<START> ' + line + ' <END>' )  

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( mar_answer ) 
tokenized_answer = tokenizer.texts_to_sequences( mar_answer ) 

length_list = list()
for token_seq in tokenized_answer:
    length_list.append( len( token_seq ))
max_output_length = np.array( length_list ).max()

padded_answer = preprocessing.sequence.pad_sequences( tokenized_answer , maxlen=max_output_length, padding='post' )
decoder_input_data = np.array( padded_answer )

answer_word_dict = tokenizer.word_index
num_answer_tokens = len( answer_word_dict )+1


# In[ ]:


decoder_target_data = list()
for token_seq in tokenized_answer:
    decoder_target_data.append( token_seq[ 1 : ] ) 
    
padded_answer = keras.preprocessing.sequence.pad_sequences( decoder_target_data , maxlen=max_output_length, padding='post' )
onehot_mar_lines = utils.to_categorical( padded_answer , num_answer_tokens )
decoder_target_data = np.array( onehot_mar_lines )


# In[ ]:


encoder_inputs = tf.keras.layers.Input(shape=( None , ))
encoder_embedding = tf.keras.layers.Embedding( num_question_tokens, 256 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 128 , return_state=True  )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
decoder_embedding = tf.keras.layers.Embedding( num_answer_tokens, 256 , mask_zero=True) (decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM( 128 , return_state=True , return_sequences=True)
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( num_answer_tokens , activation=tf.keras.activations.softmax ) 
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

model.summary()


# In[ ]:


model.fit([encoder_input_data , decoder_input_data], decoder_target_data, batch_size=250, epochs=500 )
model.save('model_chatbot.h5')


# In[ ]:


def make_inference_models():
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 128 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 128 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model


# In[ ]:


def str_to_tokens( sentence : str ):
    sentences = replace_words(sentence)
    words = sentences.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( question_word_dict[ word ] ) 
    return keras.preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=max_input_length , padding='post')


# In[ ]:


model = tf.keras.models.load_model('model_chatbot.h5')
enc_model , dec_model = make_inference_models()

for epoch in range( encoder_input_data.shape[0] ):
    states_values = enc_model.predict( str_to_tokens( input( 'Enter sentence : ' ) ) )
    #states_values = enc_model.predict( encoder_input_data[ epoch ] )
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = answer_word_dict['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in answer_word_dict.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word )
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
            stop_condition = True
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 

    print( decoded_translation )


# In[ ]:




