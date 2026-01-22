import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

train_clean = pd.read_csv("../data/train/train_clean.csv")
val_clean = pd.read_csv("../data/train/val_clean.csv")
train_full = pd.read_csv("../data/train/train_full.csv")
test_clean = pd.read_csv("../data/test/test_clean.csv")

vectorizer = TfidfVectorizer(
    ngram_range = (1,2), #get also phrases
    max_df= 0.9, #drop common words
    min_df= 5, #drop words that are too rare
    stop_words="english", #remove english stop words(a the etc)
    sublinear_tf= True #prevents repeated words or long texts from overpowering
)

vectorizer.fit(train_clean['full_text'])

X_train = vectorizer.transform(train_clean['full_text'])
X_val = vectorizer.transform(val_clean['full_text'])
y_train = train_clean['label']
y_val = val_clean['label']

model = MultinomialNB()

model.fit(X_train,y_train)

y_pred = model.predict(X_val)

print(classification_report(y_val,y_pred))
print(confusion_matrix(y_val,y_pred))
