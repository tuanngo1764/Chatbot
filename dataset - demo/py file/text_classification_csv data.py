#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[2]:


import pandas as pd
file_name = "all_data.csv"
df = pd.read_csv(file_name)
questions = df['question']
answers = df['answer']
labels = df['label']
print(labels.value_counts().index.tolist())


# In[3]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
label_ls = ' '.join(list(labels))
label_wc = WordCloud(width = 512, height = 512).generate(label_ls)
plt.figure(figsize = (10,7), facecolor = 'k')
plt.imshow(label_wc)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[4]:


import re 
from nltk.tokenize import word_tokenize
def replace_words(text): 
    LatinChar = '[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]' 
    SpecialAndSpaceChar = '[/^$*+?#!@{}&\n\t\f\r]'
    #Xử lý lọc dữ liệu
    deleteLatin = re.sub(LatinChar, '', text).strip()
    text = re.sub(SpecialAndSpaceChar, '', deleteLatin).strip()
    return text
questions = questions.apply(replace_words)
dataFrame = pd.DataFrame(df)
x = questions
y = dataFrame['label']


# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curves(estimator, title, xTest, yTest, cv, yLim = (.8, 1.01), train_sizes=np.linspace(.1, 1.0, 7)):
    plt.figure(figsize=(10,8))
    plt.title(title)
    # Chú thích tên của trục X, Y
    plt.xlabel("Training examples") 
    plt.ylabel("Score")
    # Thiêt lập giới hạn giá trị của trục Y
    plt.ylim(*yLim)
    
    #chia dữ liệu
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, xTest, yTest, cv = cv, train_sizes = train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.legend(loc="best")
    
    return plt


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.9)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

model = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('SVC', SVC(kernel = 'linear'))])
model = model.fit(X_train, y_train)
predicted = model.predict(X_test)
print(predicted[:5])
print(y_test[:5])
print(accuracy_score(y_test, predicted))


# In[7]:


title = "Learning Curves (SVC __ kernel = linear)"
estimator = SVC(kernel = 'linear')
xTest = tfidf_transformer.fit_transform(count_vect.fit_transform(x))
yTest = y
cv = ShuffleSplit(n_splits = 10, test_size = 0.1, random_state = 0)
plot_learning_curves(estimator, title, xTest, yTest, cv)
plt.show()


# In[8]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

model = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('LogisticRegression', LogisticRegression())])
model = model.fit(X_train, y_train)
predicted = model.predict(X_test)
print(predicted[:5])
print(y_test[:5])
print(accuracy_score(y_test, predicted))


# In[9]:


title = "Learning Curves (Logistic Regression)"
estimator = LogisticRegression()
xTest = tfidf_transformer.fit_transform(count_vect.fit_transform(x))
yTest = y
cv = ShuffleSplit(n_splits = 10, test_size = 0.1, random_state = 0)
plot_learning_curves(estimator, title, xTest, yTest, cv)
plt.show()


# In[10]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

model = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('MultiNB', MultinomialNB())])
model = model.fit(X_train, y_train)
predicted = model.predict(X_test)
print(predicted[:5])
print(y_test[:5])
print(accuracy_score(y_test, predicted))

