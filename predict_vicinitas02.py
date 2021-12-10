# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:54:36 2019

@author: User
"""

#import gensim.downloader as api
#from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
#model = Word2Vec.load('model_google.model')


"""Read data from csv file"""

import csv
#import random

'''
textual_data = []
textual_insult = []
test_data = []
test_insult = []
'''
predict_data = []

record = 5000

'''
with open('clean_dataset.csv', 'rt', encoding = 'latin1') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        #if ((row['Language']=='English') and (len(textual_data) < 10)):
        if ((random.randint(1,6) > 1) and (len(textual_data) < record)):
            textual_data.append(row['Comment'])
            textual_insult.append(int(row['Insult']))
'''

#data from vicinitas
with open('notmygovernment.csv', 'rt', encoding = 'latin1') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        '''if ((row['Language']=='English') and (len(textual_data) < 10)):'''
        if ((row['Language']=='English') and (len(predict_data) < 2080)):
            predict_data.append(row['Text'])
            
            
"""
print("len(texual) =", len(textual_data))
print("len(test) =", len(test_data))
"""

"""
print("textual_data({})".format(len(textual_data)))
for i in range(len(textual_data)):
    print("textual_data[{}] =".format(i), textual_data[i], '\n')
"""




"""Pre-Processing Process for Text Classification"""

from contractions import CONTRACTION_MAP #import dictionary from other file in same directory
import re
import nltk
import string
from nltk.stem import WordNetLemmatizer

#nltk.download('stopwords')
#nltk.download('punkt')

stopword_list = nltk.corpus.stopwords.words('english')
wnl = WordNetLemmatizer()


#tokenizes and removes any extraneous whitespace from the tokens
def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens


#function for expanding contractions, expand word in sentences(using regular expression)
def expand_contractions(text, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
    



from pattern.en import tag
from nltk.corpus import wordnet as wn


# Annotate text tokens with POS tags
def pos_tag_text(text):
    
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    tagged_text = tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text
    

# lemmatize text based on POS tags    
def lemmatize_text(text):
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word                     
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text
    

def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
    
    
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

    

def normalize_corpus(corpus, tokenize=True):
    normalized_corpus = [] 
    
    for text in corpus:
        text = expand_contractions(text, CONTRACTION_MAP)
        #print('expand:', text)
        text = lemmatize_text(text)
        #print('lemma:', text)
        text = remove_special_characters(text)
        #print('specail:', text)
        text = remove_stopwords(text)
        #print('stopwords:', text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)
        #print('\n')
        
    return normalized_corpus
    


'''normalize_data = normalize_corpus(textual_data[:len(textual_data)])'''
predict_normalize = normalize_corpus(predict_data[:len(predict_data)])

print("[textual_data] read {} tweet form csv file".format(len(predict_data)))
print("[normalize_data] nomalize {} tweet from textual data\n".format(len(predict_normalize)))


for i in range(len(predict_normalize)):
    print("textual_data[{}]   =".format(i+1), predict_data[i])
    print("predict_normalize[{}] =".format(i+1), predict_normalize[i], '\n')

'''
for i in range(len(normalize_data[3])):    
    try:
        vector.append(model[normalize_data[3][i]])
        print(normalize_data[3][i],'\n',model[normalize_data[3][i]])
    except:
        print("don't have this word in vocab\n")
'''

#wv = api.load('word2vec-google-news-300')
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True)
#average vector

def average_vector(words, model):
    vector = np.zeros((300),dtype="float64")
    vocabulary = set(model.index2word)
    nwords = 0
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1
            vector = np.add(vector, model[word])
    if nwords:
        vector = np.divide(vector, nwords)
    
    return vector

#vector = average_vector(normalize_data[3], model)

#feature_vector = []
#test_feature_vector = []
predict_vector = []

'''
for i in range(0,len(normalize_data)):
    vector = average_vector(normalize_data[i], model)
    feature_vector.append(vector)
'''
for i in range(0,len(predict_normalize)):
    vector = average_vector(predict_normalize[i], model)
    predict_vector.append(vector)
    print('vector of tweet{} is {}'.format(i+1, vector))


from sklearn import svm

#Add list to vector array
arrayX1 = np.array([feature_vector[0]])
for i in range(1, record):
    arrayX2 = np.append(arrayX1, [feature_vector[i]], axis=0)
    arrayX1 = arrayX2
    
#Add list to result array
arrayY1 = np.array([textual_insult[0]])
for i in range(1, record):
    arrayY2 = np.append(arrayY1, [textual_insult[i]], axis=0)
    arrayY1 = arrayY2
    
#Custom kernel
def my_kernel(X,Y):
    M = np.identity(300, dtype = float)
    return np.dot(np.dot(X, M), Y.T)

#Fit Model
clf = svm.SVC(kernel=my_kernel)
clf.fit(arrayX2, arrayY2)
#print("Classification is completed.")

#Add list to predict vector array
array_predict=np.array([predict_vector[0]])
for i in range(1, len(predict_vector)):
    array_predict2 = np.append(array_predict, [predict_vector[i]], axis=0)
    array_predict = array_predict2

prediction = clf.predict(array_predict2)

#save prediction to file
with open('predict_notmygovernment.csv', mode='w', encoding = 'latin1') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Tweet', 'Label'])
    for i in range(len(predict_data)):
        csv_writer.writerow([predict_data[i], prediction[i]])

print("success!!!\n")



'''

from sklearn import metrics
#import numpy as np
import pandas as pd
from collections import Counter

#test_insult
#prediction

ac = Counter(test_insult)                     
pc = Counter(prediction)  

"""
print ('Actual counts:', ac.most_common())
print ('Predicted counts:', pc.most_common())
"""      
        
cm = metrics.confusion_matrix(y_true=test_insult,
                         y_pred=prediction,
                         labels=[1, 0])
print (pd.DataFrame(data=cm, 
                   columns=pd.MultiIndex(levels=[['Predicted:'],
                                                 ['bully', 'not bully']], 
                                         labels=[[0,0],[0,1]]), 
                   index=pd.MultiIndex(levels=[['Actual:'],
                                               ['bully', 'not bully']], 
                                       labels=[[0,0],[0,1]])))


positive_class = 1

accuracy = np.round(metrics.accuracy_score(y_true=test_insult, y_pred=prediction),2)
print ('Accuracy:', accuracy)                                      


precision = np.round(metrics.precision_score(y_true=test_insult,
                                             y_pred=prediction,
                                             pos_label=positive_class),2)
print ('Precision:', precision)


recall = np.round(metrics.recall_score(y_true=test_insult,
                                 y_pred=prediction,
                                 pos_label=positive_class),2)
print ('Recall:', recall)


f1_score = np.round(metrics.f1_score(y_true=test_insult,
                                 y_pred=prediction,
                                 pos_label=positive_class),2)
print ('F1 score:', f1_score)

'''




  

