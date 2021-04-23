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
from nltk.corpus import stopwords
import preprocessor as p
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
def main():
    pass

if __name__ == '__main__':
    main()

TweetID_file=pd.read_csv('yelp_labelled.txt', sep="\t", header=None, names=['text','category'])
print(TweetID_file.head())

corpus=pd.DataFrame(TweetID_file,columns= ['text','category'])


X_array=list(corpus['text'])
Y_array=list(corpus['category'])

print(X_array)

setoftweets=[]

maintain_sentiments_dict=dict()
stopwords_english = stopwords.words('english')
total=len(corpus['text'])
for j in range(0,total):
    each_element=str(corpus['text'][j])

    temp_processed_tweet=each_element.split()
    tweets_clean=[]
    for word in temp_processed_tweet:
        if (word not in stopwords_english):  # remove stopwords
            tweets_clean.append(word)
    string = ' '.join(tweets_clean)
    setoftweets.append(string)


X_train, X_test, y_train, y_test = train_test_split(X_array, Y_array, test_size = 0.25, random_state=42)

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
#tfidf = CountVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
features = tfidf.fit_transform(X_train)
df=pd.DataFrame(
    features.todense(),
    columns=tfidf.get_feature_names()
)
X_test=tfidf.transform(X_test)

logreg = LogisticRegression(C=1)
logreg.fit(features, y_train)

print ("Accuracy is %s" % ( accuracy_score(y_test, logreg.predict(X_test))))

Y_predict=logreg.predict(X_test)
get_confusion_matrix=confusion_matrix(y_test, Y_predict, labels=[0,1])
print(get_confusion_matrix)

target=[0,1]
print(classification_report(y_test, Y_predict))