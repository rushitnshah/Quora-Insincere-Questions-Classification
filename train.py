import os
import re
import gc
import seaborn as sns

#Linear Algebra
import numpy as np # linear algebra
from numpy.random import random_sample
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from collections import Counter

#For ML models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score, precision_score, recall_score
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#To deal with imbalanced data
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RandomUnderSampler

#Deep learning library for RNN
from keras.layers import Input
from keras import Model
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import Callback

#For preprocessing text data
import nltk
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()
from string import punctuation

#To save variables/models
import pickle

#Import Data
df = pd.read_csv('train.csv')

def clean_review(review_col):
    review_corpus=[]
    stops = set(stopwords.words("english"))
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        word_token = word_tokenize(str(review).lower())
        review=[lemma.lemmatize(w) for w in word_token if w not in stops]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus

df['clean_question']=clean_review(df['question_text'].values)

y_train = df['target'].values
X_train_text = df['clean_question'].values

#Split into training (70%), validation (15%) and test (15%) data 
X_train_text, X_val_text, y_train, y_val = train_test_split(X_train_text, y_train, test_size=0.3)
X_val_text, X_test_text,y_val,y_test = train_test_split(X_val_text, y_val, test_size=0.5)

#Parameters to preprocess text data
num_unique_word = 166289 
MAX_QUESTION_LEN=125 #max allowable words in a question
MAX_FEATURES = num_unique_word #ceil on the number of unique words from courpus to use
MAX_WORDS = MAX_QUESTION_LEN #max allowable words in a question
tokenizer = Tokenizer(num_words=MAX_FEATURES) #tokenize training data
tokenizer.fit_on_texts(list(X_train_text))
X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(X_test_text)

X_train = sequence.pad_sequences(X_train, maxlen=MAX_WORDS)
X_val = sequence.pad_sequences(X_val, maxlen=MAX_WORDS)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_WORDS)

class BaselineModel:
    def __init__(self,type='majority'):
        self.type = type
        print("Instance of {0} classifier".format(self.type))
    def fit(self, X , y):
        if self.type=='majority':
            counts = Counter(y)
            freq_max = 0
            for label, freq in counts.items():
                if freq>freq_max:
                    freq_max = freq
                    self.label = label
        elif self.type=='random':
                self.label = 0
                
    def predict(self, X):
        predictions = np.zeros((X.shape[0],1))
        if self.type == 'majority':
            predictions = np.ones((X.shape[0],1)) * self.label
        elif self.type == 'random':
            predictions = random_sample((X.shape[0],1))
            predictions[predictions>=0.5]=1
            predictions[predictions<0.5]=0
        
        return predictions

#Functions to prepare word embeddings
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
    
def get_embed_mat(EMBEDDING_FILE, max_features,embed_dim):
    # word vectors
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf8'))
    # embedding matrix
    word_index = tokenizer.word_index
    num_words = min(max_features, len(word_index) + 1)
    all_embs = np.stack(embeddings_index.values()) #for random init
    embedding_matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        if i >= max_features:       #use only the top 125 words as features
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    max_features = embedding_matrix.shape[0]
    
    return embedding_matrix

#gloveEmbed is the embedding matrix contatining a 300-dimensional vector for each of the 137803 unqiue words in
#our vocabulary
gloveEmbed = get_embed_mat('glove.840B.300d.txt', MAX_FEATURES, 300)
word_index = tokenizer.word_index

#Training baselines models
# 0 (sincere questions) is the majority class
modelMaj = BaselineModel(type='majority')
modelMaj.fit(X_train, y_train)

modelRand = BaselineModel(type='random')
modelRand.fit(X_train, y_train)

modelLR  = LogisticRegression(random_state=0, solver='lbfgs',class_weight="balanced",C=0.1)
modelLR.fit(X_train, y_train)


modelRF = RandomForestClassifier(criterion='gini', max_depth=10, class_weight='balanced')
modelRF.fit(X_train, y_train)

lstm_out = 200
modelLSTM = Sequential()
embedding_layer = Embedding(len(word_index) + 1,300,weights=[gloveEmbed],input_length=MAX_WORDS,trainable=False)
modelLSTM.add(embedding_layer)
modelLSTM.add(LSTM(lstm_out, dropout_U = 0.4, dropout_W = 0.4))
modelLSTM.add(Dense(1,activation='sigmoid'))
modelLSTM.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(modelLSTM.summary())
modelLSTM.fit(X_train, y_train,epochs=2, batch_size=1024, verbose=1,callbacks=[metric], validation_data = (X_val,y_val),class_weight="balanced")

#Balancing data
ennus = RandomUnderSampler(random_state=0, sampling_strategy='majority')
X_train_bal, y_train_bal = ennus.fit_resample(X_train, y_train)
X_val_bal, y_val_bal = ennus.fit_resample(X_val, y_val)
X_test_bal, y_test_bal  = ennus.fit_resample(X_test , y_test)

y_total    =np.concatenate((y_train,y_val,y_test), axis=0)
y_total_bal=np.concatenate((y_train_bal,y_val_bal,y_test_bal), axis=0)
n_rows = y_total.shape[0]
n_rows_bal = y_total_bal.shape[0]
n_insincere = len(y_total[y_total==1])
n_insincere_bal = len(y_total_bal[y_total_bal==1])

label_repart = pd.DataFrame(data={"" :[n_rows - n_insincere, n_insincere]}, index = [str(n_rows - n_insincere) + ' sincere questions', str(n_insincere) + ' insincere question'])
label_repart.plot(kind='pie', title='Insincere Examples % (Before Undersampling) ' + str(round(n_insincere / n_rows,2)*100) + "%", subplots=True, figsize=(8,8))
label_repart = pd.DataFrame(data={"" :[n_rows_bal - n_insincere_bal, n_insincere_bal]}, index = [str(n_rows_bal - n_insincere_bal) + ' sincere questions', str(n_insincere_bal) + ' insincere question'])
label_repart.plot(kind='pie', title='Insincere Examples % (After Undersampling) ' + str(round(n_insincere_bal / n_rows_bal,2)*100) + "%", subplots=True, figsize=(8,8))
    
# 0 (sincere questions) is the majority class
modelMaj_bal = BaselineModel(type='majority')
modelMaj_bal.fit(X_train_bal, y_train_bal)

# 0 (sincere questions) is the majority class
modelRand_bal = BaselineModel(type='random')
modelRand_bal.fit(X_train, y_train)

#Using best parameters for final model
modelLR_bal  = LogisticRegression(random_state=0, solver='lbfgs',class_weight="balanced",C=0.001)
modelLR_bal.fit(X_train_bal, y_train_bal)

modelRF_bal = RandomForestClassifier(criterion='gini', max_depth=16, class_weight='balanced')
modelRF_bal.fit(X_train_bal, y_train_bal)

#Training LSTM with best dropout rates dropU=0.2, dropW=0.2
lstm_out = 200
modelLSTM_bal = Sequential()
embedding_layer = Embedding(len(word_index) + 1,300,weights=[gloveEmbed],input_length=MAX_WORDS,trainable=False)
modelLSTM_bal.add(embedding_layer)
modelLSTM_bal.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
modelLSTM_bal.add(Dense(1,activation='sigmoid'))
modelLSTM_bal.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
modelLSTM_bal.fit(X_train_bal, y_train_bal,epochs=2, batch_size=1024, verbose=1,callbacks=[metric], validation_data = (X_val_bal,y_val_bal))


filename = 'modelMaj'
outfile = open(filename,'wb')
pickle.dump(modelMaj_bal,outfile)
outfile.close()
filename = 'modelRand'
outfile = open(filename,'wb')
pickle.dump(modelRand_bal,outfile)
outfile.close()
filename = 'modelLR'
outfile = open(filename,'wb')
pickle.dump(modelLR_bal,outfile)
outfile.close()
filename = 'modelRF'
outfile = open(filename,'wb')
pickle.dump(modelRF_bal,outfile)
outfile.close()
filename = 'modelLSTM'
outfile = open(filename,'wb')
pickle.dump(modelLSTM_bal,outfile)
outfile.close()
filename = 'modelMaj_bal'
outfile = open(filename,'wb')
pickle.dump(modelMaj_bal,outfile)
outfile.close()
filename = 'modelRand_bal'
outfile = open(filename,'wb')
pickle.dump(modelRand_bal,outfile)
outfile.close()
filename = 'X_test_bal'
outfile = open(filename,'wb')
pickle.dump(X_test_bal,outfile)
outfile.close()
filename = 'X_val_bal'
outfile = open(filename,'wb')
pickle.dump(X_val_bal,outfile)
outfile.close()
filename = 'y_val_bal'
outfile = open(filename,'wb')
pickle.dump(y_val_bal,outfile)
outfile.close()
filename = 'y_test_bal'
outfile = open(filename,'wb')
pickle.dump(y_test_bal,outfile)
outfile.close()
filename = 'modelLR_bal'
outfile = open(filename,'wb')
pickle.dump(modelLR_bal,outfile)
outfile.close()
filename = 'modelRF_bal'
outfile = open(filename,'wb')
pickle.dump(modelRF_bal,outfile)
outfile.close()
filename = 'modelLSTM_bal'
outfile = open(filename,'wb')
pickle.dump(modelLSTM_bal,outfile)
outfile.close()
