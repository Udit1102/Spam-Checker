from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import preprocess_text
import pandas as pd
import numpy as np

#loading the data

df = pd.read_csv("sms_spam_collection.tsb", sep = "\t")
x = df.iloc[:,1]
y = df.iloc[:,0]=="spam"

#splitting the data 

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#time to build the model

vec = CountVectorizer(max_df=0.5, min_df=1, ngram_range=(1,1), preprocessor=preprocess_text)
x_train_vec = vec.fit_transform(x_train)
x_test_vec = vec.transform(x_test)

clf = MultinomialNB(alpha=0.5)
clf.fit(x_train_vec, y_train)
y_pred = clf.predict(x_test_vec)
print(f"accuracy {accuracy_score(y_test, y_pred)}")

'''
#building the pipeline for hyper parameters tuning

pipeline2 = Pipeline([("vec", CountVectorizer()), ("clf", MultinomialNB())])

#tuning hyper parameters

parameters = {"vec__ngram_range": [(1,1), (1,2)], "vec__max_df": (0.25, 0.5,0.75,1.0), "vec__min_df": (1,2), "clf__alpha": (0.01,0.1,0.5, 0.25, 0.75, 1.0)}
grid_search = GridSearchCV(pipeline2, parameters, scoring= "accuracy", cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)

#print("best parameters", grid_search.best_params_)
#evaluating the model after tuning

best_model = grid_search.best_estimator_
test_score = best_model.score(x_test, y_test)
#print(f"test score : {test_score}")
'''

#finally making the predictions for user input

c = 0
while(c == 0):
	test_text = input("please enter your message that you want to check for spam or not, enter exit to end")
	if test_text == "exit":
		print("Good Day")
		break
	test_vector = vec.transform([test_text])
	predictions = clf.predict(test_vector)
	print("You have been spammed" if predictions else "It is not a spam")