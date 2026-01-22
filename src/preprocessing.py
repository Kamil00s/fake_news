import pandas as pd
import string
from helpers import merge_text_title
from sklearn.model_selection import train_test_split


train_df = pd.read_csv("../data/train/train_data.csv")
test_df = pd.read_csv("../data/test/test_no_label.csv")


train_df['full_text'] = train_df.apply(
    lambda row: merge_text_title(row['title'],row['text']),
    axis=1
)

test_df['full_text'] = test_df.apply(
    lambda row: merge_text_title(row['title'],row['text']),
    axis=1
)

train_df['full_text'] = train_df['full_text'].str.lower()

test_df['full_text'] = test_df['full_text'].str.lower()

train_df['label'] = train_df['label'].apply(lambda x: 1 if x == "real" else 0)

test_df['label'] = test_df['label'].apply(lambda x: 1 if x == "real" else 0)

train_df['full_text'] = train_df['full_text'].str.replace(f"[{string.punctuation}]", "",regex=True)

test_df['full_text'] = test_df['full_text'].str.replace(f"[{string.punctuation}]", "",regex=True)

train_data, val_data = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['label'],
    random_state=42
)

train_data.to_csv("../data/train/train_clean.csv", index=False)
val_data.to_csv("../data/train/val_clean.csv", index=False)
train_df.to_csv("../data/train/train_full.csv",index=False)
test_df.to_csv("../data/test/test_clean.csv",index=False)


