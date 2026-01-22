import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


train_clean = pd.read_csv("../data/train/train_clean.csv")
val_clean = pd.read_csv("../data/train/val_clean.csv")
train_full = pd.read_csv("../data/train/train_full.csv")
test_clean = pd.read_csv("../data/test/test_clean.csv")

vectorizer = TfidfVectorizer()

vectorizer.fit(train_full['full_text'])

X_train = vectorizer.transform(train_full['full_text'])
X_test = vectorizer.transform(test_clean['full_text'])
y_train = train_full['label']
y_test = test_clean['label']

model = MultinomialNB()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
