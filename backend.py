import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class SpamDetector:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None
        self.accuracy = 0.0
        self._train_model()

    def _train_model(self):
        df = pd.read_csv(self.data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            df["message"], df["label"], test_size=0.2, random_state=42
        )

        pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])

        pipeline.fit(X_train, y_train)
        self
