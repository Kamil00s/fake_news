# Fake News Detection with TF-IDF and Naive Bayes

## 📌 Project Overview
This project detects fake news articles using classical NLP techniques.
The model is trained on article titles + content using TF-IDF vectorization
and a Multinomial Naive Bayes classifier.

## 🧠 Motivation
Fake news is a growing problem in online media. This project explores
how traditional machine learning methods perform on this task.

## 📂 Dataset
Source: Kaggle – Fake News Detection by Bhavik Jikadara
link: https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=true.csv
Features used:
- title
- text  
Labels:
- real (1)
- fake (0)

## ⚙️ Pipeline
1. Merge title and text
2. Text normalization (lowercase, punctuation removal)
3. TF-IDF vectorization (unigrams + bigrams)
4. Naive Bayes classification
5. Evaluation on validation set

## 📊 Results
Accuracy: ~94% (base model)


## 🛠️ Tech Stack
- Python
- pandas
- scikit-learn
- matplotlib / seaborn

## ▶️ How to Run
```bash
pip install -r requirements.txt
python src/predict.py

