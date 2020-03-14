from nlp_disaster.methods.formatting import clean_stopwords, clean_puncts
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import json
import os
import pandas as pd
from datetime import datetime

cur_date = datetime.utcnow().strftime("%Y%m%d")

DATA_PATH = "./data/"
INPUT_PATH = "./data/input/"

OUT_PATH = "./out/"
if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)

with open(f"{DATA_PATH}formatting.json") as infile:
    FORMATTING = json.load(infile)

train_df = pd.read_csv(f"{INPUT_PATH}train.csv").fillna("")
test_df = pd.read_csv(f"{INPUT_PATH}test.csv").fillna("")
sample_submission = pd.read_csv(f"{INPUT_PATH}sample_submission.csv")

if __name__ == "__main__":
    train_df["no-punct"] = train_df["text"].apply(
        lambda x: clean_puncts(x, FORMATTING)
    )
    test_df["no-punct"] = test_df["text"].apply(
        lambda x: clean_puncts(x, FORMATTING)
    )

    count_vectorizer = feature_extraction.text.CountVectorizer()
    train_vector = count_vectorizer.fit_transform(train_df["no-punct"])
    test_vector = count_vectorizer.transform(test_df["no-punct"])

    clf = linear_model.RidgeClassifier()
    clf.fit(train_vector, train_df["target"])

    sample_submission["target"] = clf.predict(test_vector)
    sample_submission.to_csv(
        f"{OUT_PATH}submission - {cur_date}.csv", index=False)
