#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      lenovo
#
# Created:     22-04-2021
# Copyright:   (c) lenovo 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import pandas as pd
import csv
import re
import nltk
import networkx as nx
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as PS
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import preprocessor as p
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc

#prepare targets
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

def select_features(X_train, y_train, X_test):

	fs = SelectKBest(score_func=chi2, k=400)
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)

	return X_train_fs, X_test_fs, fs

def main():
    pass

if __name__ == '__main__':
    main()

TweetID_file=pd.read_csv('yelp_labelled.txt', sep="\t", header=None, names=['text','category'])


corpus=pd.DataFrame(TweetID_file,columns= ['text','category'])


X_array=list(corpus['text'])
Y_array=list(corpus['category'])


setoftweets=[]

maintain_sentiments_dict=dict()
stopwords_english = stopwords.words('English')
total=len(corpus['text'])
porter=PS()


for j in range(0,total):
    each_element=str(corpus['text'][j])

    temp_processed_tweet=each_element.split()

    tweets_clean=[]

    for eachword in temp_processed_tweet:
                eachword=porter.stem(eachword)
                if (eachword not in stopwords_english):  # remove stopwords
                    tweets_clean.append(eachword)
    string = ' '.join(tweets_clean)
    setoftweets.append(string)

#newsetofwords=getnewst(setoftweets)

X_array=list(setoftweets)

X_train, X_test, y_train, y_test = train_test_split(X_array, Y_array, test_size = 0.25, random_state=42)

#tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
tfidf = CountVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
X_train = tfidf.fit_transform(X_train)


df=pd.DataFrame(
    X_train.todense(),
    columns=tfidf.get_feature_names()


)

print(tfidf.get_feature_names())
X_test=tfidf.transform(X_test)
##
###X_train_enc, X_test_enc = X_train, X_test
### prepare output data
##y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
##
##
### feature selection
##X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)


logreg = LogisticRegression(C=1)
logreg.fit(X_train, y_train)

print ("Accuracy is %s" % ( accuracy_score(y_test, logreg.predict(X_test))))

Y_predict=logreg.predict(X_test)
get_confusion_matrix=confusion_matrix(y_test, Y_predict, labels=[0,1])
print(get_confusion_matrix)

target=[0,1]
print(classification_report(y_test, Y_predict))