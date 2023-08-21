# Reference Code for my Project
# Source Code:
# https://github.com/jacksonyuan-yt/youtube-comments-spam-classifier/blob/master/youtube-comments-spam-classifier.ipynb

import pandas as pd
import zipfile
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix, classification_report


# Import dataset files:
z = zipfile.ZipFile("YouTubeSpam.zip")
Psy = pd.read_csv(z.open("Youtube01-Psy.csv"))
Katy = pd.read_csv(z.open("Youtube02-KatyPerry.csv"))
LMFAO = pd.read_csv(z.open("Youtube03-LMFAO.csv"))
Eminem = pd.read_csv(z.open("Youtube04-Eminem.csv"))
Shakira = pd.read_csv(z.open("Youtube05-Shakira.csv"))

data = pd.concat([Psy, Katy, LMFAO, Eminem, Shakira])
data.drop(["COMMENT_ID", "DATE", "AUTHOR"], axis=1, inplace=True)
print("\nImported Data:")
print(data.info())
print()


# Splitting dataset into train/test sets
X_train, X_test, y_train, y_test = train_test_split(data["CONTENT"], data["CLASS"])

# Tokenizing comments in training set and applying TF-IDF vectorizer on training set
tfidf_vect = TfidfVectorizer(use_idf=True, lowercase=True)
X_train_tfidf = tfidf_vect.fit_transform(X_train)
print("Tokenized Data:")
print(X_train_tfidf.shape)
print()

# Training the multinomial Naive Bayes model
model = MultinomialNB()
print("Multinomial Naive Bayes Model:")
print(model.fit(X_train_tfidf, y_train))
print()

# Generate predictions on test set
X_test_tfidf = tfidf_vect.transform(X_test)
predictions = model.predict(X_test_tfidf)

# Generate model performance metrics
print("Performance Metrics:")
print(confusion_matrix(y_test, predictions))
print()

print("Classification Report:")
print(classification_report(y_test, predictions))
print()

print("Model Performance Metrics:")
print(model.score(X_test_tfidf, y_test))
print()

# Exporting the model and TF-IDF vectorizer
with open("model.pkl", "wb") as model_file:
  pickle.dump(model, model_file)

with open("tfidf-vect.pkl", "wb") as tfidf_vect_file:
  pickle.dump(tfidf_vect, tfidf_vect_file)