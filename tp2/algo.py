import streamlit as sl
import pandas as pd
import numpy as np
import joblib as jl
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from stop_words import get_stop_words
from sklearn.svm import SVC

df = pd.read_csv("data/labels.csv")
#df
#df.isna().sum()

clf = Pipeline([('TfidfVectorizer', TfidfVectorizer(stop_words=get_stop_words('en'))), 
                ('OneVsRestClassifier', OneVsRestClassifier(SVC(kernel='linear', probability=True)))])
clf = clf.fit(X=df['tweet'], y=df['class'])

jl.dump(clf, 'model.joblib')