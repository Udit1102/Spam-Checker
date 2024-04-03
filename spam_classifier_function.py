from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import preprocess_text
import pandas as pd
from sklearn.pipeline import Pipeline

#loading the data

df = pd.read_csv("sms_spam_collection.tsb", sep = "\t")
x = df.iloc[:,1]
y = df.iloc[:,0]=="spam"

#splitting the data 

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#time to build the model with pipeline

pipeline = Pipeline([("vec", CountVectorizer(preprocessor=preprocess_text, max_df=0.5, min_df=1, ngram_range=(1,1))), ("clf", MultinomialNB(alpha=0.5))])
pipeline.fit(x_train, y_train)

score = pipeline.score(x_test, y_test)
#print(score)

def predicter(test_text, pipeline):
	prediction = pipeline.predict(test_text)
	return "You have been spammed" if prediction else "It is not a spam"

