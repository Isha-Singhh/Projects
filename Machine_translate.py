#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense


# In[26]:


lines = pd.read_table("mar.txt",names = ['mar','random'])
lines.head()


# In[27]:


lines.reset_index(inplace = True)
lines = lines.drop(['random'], axis = 1)
lines = lines.rename(columns={"index":"eng"})


# In[4]:


type(lines)
lines.head()


# In[28]:


# Lowercase all the characters
lines.eng = lines.eng.apply(lambda x: x.lower())
lines.mar = lines.mar.apply(lambda x: x.lower())

# Remove quotes
lines.eng=lines.eng.apply(lambda x: re.sub("'", '', x))
lines.mar=lines.mar.apply(lambda x: re.sub("'", '', x))

exclude = set(string.punctuation) # Set of all special characters
# Remove all the special characters
lines.eng=lines.eng.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines.mar=lines.mar.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

# Remove all numbers from text
remove_digits = str.maketrans('', '', digits)
lines.eng=lines.eng.apply(lambda x: x.translate(remove_digits))
lines.mar = lines.mar.apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

# Remove extra spaces
lines.eng=lines.eng.apply(lambda x: x.strip())
lines.mar=lines.mar.apply(lambda x: x.strip())
lines.eng=lines.eng.apply(lambda x: re.sub(" +", " ", x))
lines.mar=lines.mar.apply(lambda x: re.sub(" +", " ", x))

# Add start and end tokens to target sequences
lines.mar = lines.mar.apply(lambda x : 'START_ '+ x + ' _END')


# In[19]:


lines.sample(10)


# In[29]:


#Vocabulary of english
all_eng_words = set()
for eng in lines.eng:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

#Vocabulary of marathi
all_mar_words = set()
for mar in lines.mar:
    for word in mar.split():
        if word not in all_mar_words:
            all_mar_words.add(word)


# In[30]:


all_eng_words


# In[31]:


all_mar_words


# In[32]:


length_list = []
for l in lines.eng:
    length_list.append(len(l.split(" ")))
max_length_tar = np.max(length_list)
max_length_tar


# In[33]:


length_list = []
for l in lines.mar:
    length_list.append(len(l.split(" ")))
max_length_tar = np.max(length_list)
max_length_tar


# In[34]:


input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_mar_words))
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_mar_words)
num_encoder_tokens, num_decoder_tokens


# In[35]:


num_decoder_tokens += 1
num_decoder_tokens


# In[36]:


num_encoder_tokens += 1
num_encoder_tokens


# In[41]:


input_token_index = dict([(word, i+1) for i , word in enumerate(input_words)])
target_token_index = dict([(word, i+1) for i , word in enumerate(target_words)])


# In[42]:


reverse_input_char_index = dict(((i,word) for word, i in input_token_index.items()))
reverse_target_char_index = dict(((i,word) for word, i in target_token_index.items()))


# In[43]:


input_token_index


# In[44]:


target_token_index


# In[ ]:




