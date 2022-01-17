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
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import operator
import math
import preprocessor as p
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import normalize
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

topnum=30
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

	fs = SelectKBest(score_func=chi2, k=500)
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)

	return X_train_fs, X_test_fs, fs


def do_transform(X_test, all_features):
    print(type(X_test))
    df = pd.DataFrame(columns=all_features.keys())
    print(df)

    features_setoftweets=[]
    for anything in X_test:
        listofwords=anything.split()
        each_tweet_1=anything.split()

        for i in range(0, len(each_tweet_1)-1):
            listofwords.append(str(each_tweet_1[i]+' '+ each_tweet_1[i+1]))
        features_setoftweets.append(listofwords)



    full_featureset=[]
    for each_feature in features_setoftweets:
        everytweet_featureset=dict()
        for each_word in each_feature:

            temp_each_word=each_word.split()

            if len(temp_each_word)==1:
                    try:
                        everytweet_featureset[each_word]=all_features[each_word]
                    except:
                        continue
            if len(temp_each_word)==2:

                try:
                    everytweet_featureset[each_word]=all_features[str(each_word)]
                except:
                    continue

        df=df.append(everytweet_featureset, ignore_index=True)
        full_featureset.append(everytweet_featureset)

    df=df.replace(np.nan, 0)
    num_array= df.to_numpy()

    normed_matrix = normalize(num_array)

    return normed_matrix

def getnewst(setoftweets):

    edges=dict()
    for each_tweet in setoftweets:
        each_tweet=each_tweet.split()

        for i in range(0, len(each_tweet)-1):
            if (each_tweet[i], each_tweet[i+1]) in edges.keys():
                edges[(each_tweet[i], each_tweet[i+1])]= edges[(each_tweet[i], each_tweet[i+1])]+1

            else:
                edges[(each_tweet[i], each_tweet[i+1])]=1


    G=nx.Graph()
    for (u,v),w in edges.items():
        G.add_edge(u,v,weight=w)



    newresults_S=dict(G.degree(weight='weight'))
    newresults_k=dict(G.degree())
    clust=nx.clustering(G)
    selectivity_dict=dict()
    for each_node in G.nodes():
        selectivity_dict[each_node]=(float)((float)(newresults_S[each_node])/(float)(newresults_k[each_node]))*(math.log(newresults_k[each_node]))

    edge_score=dict()
    for (u,v) in G.edges():
        try:
            ew=edges[(u,v)]
        except:
            ew=edges[(v,u)]
        es=(float)((float)(ew)/(float)(G.degree(u)+G.degree(v)-ew))

        edge_score[(u,v)]=(float)(math.log(ew)*(float)(es))

    results_unigram = sorted(selectivity_dict.items(),key=operator.itemgetter(1),reverse=True)
    results_bigram = sorted(edge_score.items(),key=operator.itemgetter(1),reverse=True)


    feature_value_uni=dict()
    for each_unigram in selectivity_dict.keys():
        if selectivity_dict[each_unigram]>1:
            feature_value_uni[each_unigram]=selectivity_dict[each_unigram]

    print(type(feature_value_uni))

    max_val=max(feature_value_uni.values())

    feature_dict=dict()
    for k,v in feature_value_uni.items():
        feature_dict[k]=(float)(((float)(v))/(max_val))

    feature_value_uni=feature_dict

    feature_value_bi=dict()
    for (u,v) in edge_score.keys():
        if edge_score[(u,v)]>0.0:
            bigram_str=str(u+' '+v)
            feature_value_bi[bigram_str]=edge_score[(u,v)]

    max_val=max(feature_value_bi.values())

    feature_dict=dict()
    for k,v in feature_value_uni.items():
        feature_dict[k]=(float)(((float)(v))/(max_val))

    feature_value_bi=feature_dict

    all_features=dict()
    for k,v in feature_value_uni.items():
        all_features[k]=str(v)

    for k,v in feature_value_bi.items():
        all_features[k]=str(v)

    df = pd.DataFrame(columns=all_features.keys())
    print(df)

    features_setoftweets=[]
    for anything in setoftweets:
        listofwords=anything.split()
        each_tweet_1=anything.split()

        for i in range(0, len(each_tweet_1)-1):
            listofwords.append(str(each_tweet_1[i]+' '+ each_tweet_1[i+1]))
        features_setoftweets.append(listofwords)



    full_featureset=[]
    for each_feature in features_setoftweets:
        everytweet_featureset=dict()
        for each_word in each_feature:

            temp_each_word=each_word.split()

            if len(temp_each_word)==1:
                    try:
                        everytweet_featureset[each_word]=feature_value_uni[each_word]
                    except:
                        continue
            if len(temp_each_word)==2:

                try:
                    everytweet_featureset[each_word]=feature_value_bi[str(each_word)]
                except:
                    continue

        df=df.append(everytweet_featureset, ignore_index=True)

        full_featureset.append(everytweet_featureset)

    df=df.replace(np.nan, 0)
    num_array= df.to_numpy()

    normed_matrix = normalize(num_array)

    print(normed_matrix)

    return normed_matrix, all_features
def main():
    pass

if __name__ == '__main__':
    main()

TweetID_file=pd.read_csv('amazon_cells_labelled.txt', sep="\t", header=None, names=['text','category'])


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
    result=list(nltk.pos_tag(temp_processed_tweet))

    for (eachword,val) in result:
 #       if( val == 'VB' or val=='ADJ'  or val == 'ADV'):
  #          eachword=porter.stem(eachword)
            eachword=eachword.lower()
            eachword=eachword.replace(".", " ")
            eachword=eachword.replace("!", " ")
            eachword=eachword.replace(",", " ")
            eachword=eachword.replace("-", " ")

            if (eachword not in stopwords_english):  # remove stopwords

                tweets_clean.append(eachword)
    string = ' '.join(tweets_clean)
    setoftweets.append(string)

X_train, X_test, y_train, y_test = train_test_split(setoftweets, Y_array, test_size = 0.25, random_state=42)


X_train, all_features=getnewst(X_train)

X_test=do_transform(X_test, all_features)
print(abc1)


##X_array=list(setoftweets)
##
##
###tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
##tfidf = CountVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
##X_train = tfidf.fit_transform(X_train)
##
##
##
##df=pd.DataFrame(
##    X_train.todense(),
##    columns=tfidf.get_feature_names()
##)
##
##
##X_test=tfidf.transform(X_test)
##
###X_train_enc, X_test_enc = X_train, X_test
### prepare output data
##y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
##
##
### feature selection
##X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
##

logreg = LogisticRegression(C=1)
logreg.fit(X_train, y_train)

print ("Accuracy is %s" % ( accuracy_score(y_test, logreg.predict(X_test))))

Y_predict=logreg.predict(X_test)
get_confusion_matrix=confusion_matrix(y_test, Y_predict, labels=[0,1])
print(get_confusion_matrix)

target=[0,1]
print(classification_report(y_test, Y_predict))