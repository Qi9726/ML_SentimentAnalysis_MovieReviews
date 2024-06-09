import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == "__main__":

    # Open the file and read its contents into a list
    with open('myvocab.txt', 'r') as file:
        lines = file.readlines()
    vocabulary = [line.strip() for line in lines]

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    vectorizer.fit(vocabulary)

    train_path = "train.tsv"
    test_path = "test.tsv"

    traindf = pd.read_csv(train_path, sep='\t')
    testdf = pd.read_csv(test_path, sep='\t')

    traindf['review'] = traindf['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)
    testdf['review'] = testdf['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

    X_train = vectorizer.transform(traindf['review'])
    X_test = vectorizer.transform(testdf['review'])

    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(penalty='elasticnet',solver='saga',random_state=4844,C=1, l1_ratio=0.2).fit(X_train, traindf['sentiment'])
    predictions = clf.predict_proba(X_test)

    y_predict = pd.DataFrame(predictions[:,1], columns=["prob"])

    output_df = pd.concat([testdf["id"], y_predict], axis=1)
    output_df.to_csv("mysubmission.csv", sep=',', index=False)

